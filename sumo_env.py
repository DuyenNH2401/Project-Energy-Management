# sumo_env.py
# Core SUMO Gymnasium environment — inherits vehicle dynamics & map loading
# from old_env, but with CLEAN reward delegation and improved robustness.
#
# DESIGN PRINCIPLE: The reward function is NOT computed here.
# Base env ALWAYS returns reward=0.0.
# Terminal events (collision, success, stuck, etc.) are signaled via
# info["terminal_event"] — stage-specific wrappers apply their own rewards.
#
# PERFORMANCE: Uses libsumo (in-process) instead of traci (TCP socket)
# for 3-6x speedup. Falls back to libtraci if libsumo not available.

import os
import sys
import time
import random
import uuid

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config as cfg

# --- SUMO PATH SETUP ---
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# --- SUMO API SELECTION ---
# Priority: libsumo (fastest, in-process) > libtraci (C++ client) > traci (pure Python)
# NOTE: libsumo does NOT support sumo-gui on Windows.
#       libsumo may also fail with DLL errors on Windows if version mismatches.
#       If GUI rendering is needed, we fall back to libtraci/traci.
_SUMO_BACKEND = None
_SUMO_SUPPORTS_GUI = False

try:
    import libsumo as traci
    _SUMO_BACKEND = "libsumo"
    _SUMO_SUPPORTS_GUI = False  # libsumo doesn't support GUI on Windows
    print("[SUMO] Using libsumo backend (in-process, fastest)")
except (ImportError, OSError, Exception) as e:
    try:
        import libtraci as traci
        _SUMO_BACKEND = "libtraci"
        _SUMO_SUPPORTS_GUI = True
        print("[SUMO] Using libtraci backend (C++ client, fast)")
    except (ImportError, OSError, Exception):
        import traci
        _SUMO_BACKEND = "traci"
        _SUMO_SUPPORTS_GUI = True
        print("[SUMO] Using traci backend (pure Python, slowest)")


class SumoEnv(gym.Env):
    """
    Base SUMO environment with 25-dim observation and 2-dim continuous action.
    Reward is ALWAYS 0.0 — stage-specific wrappers compute ALL rewards
    including terminal rewards via info["terminal_event"].

    BACKEND: Automatically selects the fastest available SUMO API:
      libsumo (in-process) > libtraci (C++ client) > traci (Python TCP)
    """

    metadata = {"render_modes": ["human", "none"]}

    def __init__(
        self,
        render: bool = False,
        map_config: list[str] | str | None = None,
        traffic_scale: float = cfg.TRAFFIC_SCALE,
        delay: int = 20,
        max_episode_steps: int = cfg.MAX_EPISODE_STEPS,
    ) -> None:
        super().__init__()

        # --- Vehicle identity ---
        self.VEH_ID = "ego_vehicle"
        self.VTYPE_ID = "vinfast_vf9"

        # --- Physics constants (from config) ---
        self.MAX_SPEED = cfg.MAX_SPEED
        self.MAX_ACCEL = cfg.MAX_ACCEL
        self.MAX_DECEL = cfg.MAX_DECEL
        self.MAX_ELEC  = cfg.MAX_ELEC
        self.MAX_SLOPE = cfg.MAX_SLOPE
        self.MAX_DIST  = cfg.MAX_DIST
        self.TARGET_DIST = cfg.TARGET_DIST
        self.MAX_LAT_OFFSET = cfg.MAX_LAT_OFFSET
        self.LC_THRESHOLD = cfg.LC_THRESHOLD
        self.SIM_STEPS = cfg.SIM_STEPS_PER_ACTION

        # --- Environment settings ---
        # If using libsumo (no GUI support on Windows), force headless mode
        if render and not _SUMO_SUPPORTS_GUI:
            print("[WARN] libsumo does not support GUI on Windows. Forcing headless mode.")
            print("       To use GUI, install libtraci or traci: pip install libtraci")
            self.render_mode = False
        else:
            self.render_mode = render

        self.maps = ([map_config] if isinstance(map_config, str) else map_config) or cfg.MAP_CONFIG
        self.traffic_scale = traffic_scale
        self.delay = delay
        self.max_episode_steps = max_episode_steps
        self.step_count = 0

        # --- Spaces ---
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        # 25-dim observation: 9 ego + 8 surroundings + 2 leader + 6 infrastructure
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32
        )

        # --- Internal state ---
        self.label = f"env_{uuid.uuid4().hex}"
        self.last_known_dist = 0.0
        self.stuck_time = 0
        self._success = False
        self.prev_action = None
        self.prev_dist = None
        self.current_route_edges: list[str] = []
        self.drivable_edges: list[str] = []
        self._sim_running = False  # Track simulation state (libsumo has no isLoaded())

    # =========================================================================
    # OBSERVATION
    # =========================================================================
    def _veh_exists(self) -> bool:
        try:
            return self.VEH_ID in traci.vehicle.getIDList()
        except Exception:
            return False

    def _get_passenger_edges(self) -> list[str]:
        """Filter map edges that allow passenger vehicles."""
        valid = []
        for edge_id in traci.edge.getIDList():
            if edge_id.startswith(":") or edge_id.startswith("!"):
                continue
            try:
                allowed = traci.lane.getAllowed(f"{edge_id}_0")
                if not allowed or "passenger" in allowed:
                    valid.append(edge_id)
            except Exception:
                continue
        random.shuffle(valid)
        return valid

    def _get_surroundings(self) -> list[float]:
        """
        8-dim: [dist, rel_speed] for L-front, L-back, R-front, R-back.

        SUMO getNeighbors mode bitset:
          bit 0 (value 1): right side (unset = left side)
          bit 1 (value 2): ahead/leading (unset = behind/following)
        So: 0=Left-Behind, 1=Right-Behind, 2=Left-Ahead, 3=Right-Ahead
        """
        data = {
            "LF_d": 1.0, "LF_v": 0.0,
            "LB_d": 1.0, "LB_v": 0.0,
            "RF_d": 1.0, "RF_v": 0.0,
            "RB_d": 1.0, "RB_v": 0.0,
        }
        my_speed = traci.vehicle.getSpeed(self.VEH_ID)

        # Each direction gets its own explicit call with correct bitmask
        neighbors_config = [
            (2, "LF_d", "LF_v"),  # mode=0b10: Left Ahead
            (0, "LB_d", "LB_v"),  # mode=0b00: Left Behind
            (3, "RF_d", "RF_v"),  # mode=0b11: Right Ahead
            (1, "RB_d", "RB_v"),  # mode=0b01: Right Behind
        ]

        for mode, key_d, key_v in neighbors_config:
            try:
                neighbors = traci.vehicle.getNeighbors(self.VEH_ID, mode)
                if neighbors:
                    # Take the closest neighbor
                    n_id, dist = neighbors[0]
                    n_speed = traci.vehicle.getSpeed(n_id)
                    data[key_d] = min(abs(dist), self.MAX_DIST) / self.MAX_DIST
                    data[key_v] = (my_speed - n_speed) / self.MAX_SPEED
            except Exception:
                pass

        return list(data.values())

    def _get_obs(self) -> np.ndarray:
        """Build the 25-dim observation vector."""
        if not self._veh_exists():
            return np.zeros(25, dtype=np.float32)

        # --- Ego Physics (9) — ALL normalized to ~[-1, 1] ---
        velocity = traci.vehicle.getSpeed(self.VEH_ID)
        acceleration = traci.vehicle.getAcceleration(self.VEH_ID)
        elec = traci.vehicle.getElectricityConsumption(self.VEH_ID) / self.MAX_ELEC

        # Normalize velocity and acceleration
        norm_velocity = velocity / self.MAX_SPEED
        if acceleration >= 0:
            norm_accel = acceleration / self.MAX_ACCEL
        else:
            norm_accel = acceleration / self.MAX_DECEL
        norm_accel = np.clip(norm_accel, -1.0, 1.0)

        try:
            lane_idx = traci.vehicle.getLaneIndex(self.VEH_ID)
            road_id = traci.vehicle.getRoadID(self.VEH_ID)
            total_lanes = traci.edge.getLaneNumber(road_id)
            norm_lane = lane_idx / max(1, total_lanes - 1)
            slope = traci.vehicle.getSlope(self.VEH_ID) / self.MAX_SLOPE
            raw_lat_offset = traci.vehicle.getLateralLanePosition(self.VEH_ID)
            norm_lat_offset = np.clip(raw_lat_offset / self.MAX_LAT_OFFSET, -1.0, 1.0)
            # Lateral speed: useful for lane-change dynamics
            lateral_speed = traci.vehicle.getLateralSpeed(self.VEH_ID) / self.MAX_SPEED
            # Heading as sin+cos: continuous signal with NO circular ambiguity
            angle_rad = np.radians(traci.vehicle.getAngle(self.VEH_ID))
            heading_sin = np.sin(angle_rad)
            heading_cos = np.cos(angle_rad)
        except Exception:
            norm_lane = 0.0
            slope = 0.0
            norm_lat_offset = 0.0
            lateral_speed = 0.0
            heading_sin = 0.0
            heading_cos = 1.0  # cos(0) = 1.0 as default

        # --- Leader (2) ---
        try:
            leader = traci.vehicle.getLeader(self.VEH_ID, dist=self.MAX_DIST)
            if leader:
                l_dist = leader[1] / self.MAX_DIST
                l_rel_speed = (velocity - traci.vehicle.getSpeed(leader[0])) / self.MAX_SPEED
            else:
                l_dist, l_rel_speed = 1.0, 0.0
        except Exception:
            l_dist, l_rel_speed = 1.0, 0.0

        # --- Surroundings (8) ---
        surroundings = self._get_surroundings()

        # --- Infrastructure (6) ---
        try:
            lane_id = traci.vehicle.getLaneID(self.VEH_ID)
            if lane_id == "":
                speed_limit = 1.0
                can_left, can_right = 0.0, 0.0
                tls_dist, tls_state, turn_dist = 1.0, 1.0, 1.0
            else:
                speed_limit = traci.lane.getMaxSpeed(laneID=lane_id) / self.MAX_SPEED
                road_id = traci.vehicle.getRoadID(self.VEH_ID)
                num_lanes = traci.edge.getLaneNumber(road_id)
                can_left = 1.0 if lane_idx < (num_lanes - 1) else 0.0
                can_right = 1.0 if lane_idx > 0 else 0.0

                tls_data = traci.vehicle.getNextTLS(self.VEH_ID)
                if tls_data:
                    tls_dist = tls_data[0][2] / self.MAX_DIST
                    tls_state = 1.0 if tls_data[0][3].lower() == "g" else 0.0
                else:
                    tls_dist, tls_state = 1.0, 1.0

                lane_len = traci.lane.getLength(lane_id)
                lane_pos = traci.vehicle.getLanePosition(self.VEH_ID)
                turn_dist = min(lane_len - lane_pos, self.MAX_DIST) / self.MAX_DIST
        except Exception:
            speed_limit, can_left, can_right = 1.0, 0.0, 0.0
            tls_dist, tls_state, turn_dist = 1.0, 1.0, 1.0

        # 25-dim: 9 ego + 8 surroundings + 2 leader + 6 infrastructure
        obs = np.array([
            norm_velocity, norm_accel, elec, norm_lane,
            slope, norm_lat_offset, lateral_speed, heading_sin, heading_cos,
            *surroundings,
            l_dist, l_rel_speed,
            speed_limit, can_left, can_right, tls_dist, tls_state, turn_dist,
        ], dtype=np.float32)

        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    # =========================================================================
    # DISTANCE HELPERS
    # =========================================================================
    def _get_dist_to_destination(self) -> float:
        try:
            current_edge = traci.vehicle.getRoadID(self.VEH_ID)
            if current_edge.startswith(":"):
                return self.last_known_dist if self.last_known_dist else 0.0

            if current_edge in self.current_route_edges:
                idx = self.current_route_edges.index(current_edge)
                remaining = self.current_route_edges[idx:]
            else:
                return 0.0

            dist = sum(traci.lane.getLength(f"{e}_0") for e in remaining)
            dist -= traci.vehicle.getLanePosition(self.VEH_ID)
            self.last_known_dist = dist
            return dist
        except Exception:
            return 0.0

    # =========================================================================
    # RESET
    # =========================================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # Close previous SUMO instance
        # NOTE: libsumo has no isLoaded() — we track simulation state manually
        try:
            if self._sim_running:
                traci.switch(self.label)
                traci.close()
                self._sim_running = False
        except Exception:
            self._sim_running = False

        # Pick a random map
        active_map = random.choice(self.maps)
        self.step_count = 0
        self._success = False
        self.prev_action = None
        self.prev_dist = None
        self.stuck_time = 0

        # Launch SUMO
        # libsumo: always use "sumo" (headless). GUI is not supported.
        # libtraci/traci: can use "sumo-gui" if render_mode is True.
        binary = "sumo-gui" if self.render_mode else "sumo"
        cmd = [
            binary, "-c", active_map,
            "--start", "--quit-on-end",
            "--device.emissions.probability", "1.0",
            "--scale", str(self.traffic_scale),
            "--delay", str(self.delay),
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--time-to-teleport", "-1",
            "--collision.action", "warn",
        ]
        traci.start(cmd, label=self.label)
        traci.switch(self.label)
        self._sim_running = True

        # Warm up traffic
        for _ in range(random.randint(150, 300)):
            traci.simulationStep()

        # Setup vehicle type
        self._setup_vehicle_type()

        # Spawn ego vehicle on random route
        self.drivable_edges = self._get_passenger_edges()
        spawned = self._spawn_ego_vehicle()

        if not spawned:
            print("[WARN] Spawn failed — reloading map...")
            return self.reset(seed=seed, options=options)

        # Camera tracking (only when GUI is available)
        if self.render_mode and _SUMO_SUPPORTS_GUI and self._veh_exists():
            try:
                traci.gui.trackVehicle("View #0", self.VEH_ID)
                traci.gui.setZoom("View #0", 2000)
            except Exception:
                pass  # GUI commands may fail in headless mode

        obs = self._get_obs()
        return obs, {}

    def _setup_vehicle_type(self):
        """Configure the Vinfast VF9 EV type in SUMO."""
        try:
            existing = list(traci.vehicletype.getIDList())  # libsumo needs explicit list()
            source = "DEFAULT_VEHTYPE" if "DEFAULT_VEHTYPE" in existing else existing[0]

            traci.vehicletype.copy(source, self.VTYPE_ID)
            traci.vehicletype.setVehicleClass(self.VTYPE_ID, "passenger")
            traci.vehicletype.setColor(self.VTYPE_ID, (0, 255, 0))
            traci.vehicletype.setParameter(self.VTYPE_ID, "mass", "2911")
            traci.vehicletype.setLength(self.VTYPE_ID, 5.1181)
            traci.vehicletype.setEmissionClass(self.VTYPE_ID, "MMPEVEM")
            traci.vehicletype.setParameter(self.VTYPE_ID, "has.battery.device", "true")
            traci.vehicletype.setParameter(self.VTYPE_ID, "device.battery.capacity", "123000.00")
            traci.vehicletype.setParameter(self.VTYPE_ID, "device.battery.chargeLevel", "123000.00")

            # Background traffic imperfection
            for vt in existing:
                traci.vehicletype.setParameter(vt, "sigma", str(cfg.DRIVER_IMPERFECTION))
                traci.vehicletype.setParameter(vt, "impatience", str(cfg.DRIVER_IMPATIENCE))
        except Exception as e:
            print(f"[WARN] Vehicle type setup error: {e}")

    def _spawn_ego_vehicle(self) -> bool:
        """Best-effort route generation and vehicle spawning.
        
        FIX: Always generate a NEW random route per episode (removed fixed_route_edges).
        Using a fixed route caused SUMO to delete the ego vehicle at route end,
        which was then misclassified as 'vanished' instead of 'success'.
        Route diversity also improves agent generalization.
        """
        best_edges = None
        best_length = 0.0

        for _ in range(100):
            try:
                e_start = random.choice(self.drivable_edges)
                e_end = random.choice(self.drivable_edges)
                if e_start == e_end:
                    continue

                route = traci.simulation.findRoute(e_start, e_end, self.VTYPE_ID)
                if not route.edges:
                    continue

                length = sum(
                    traci.lane.getLength(f"{e}_0")
                    for e in route.edges
                    if not e.startswith(":")
                )

                if length > 1000.0:
                    best_edges = list(route.edges)  # libsumo: ensure list type
                    best_length = length
                    break
                elif length > best_length:
                    best_edges = list(route.edges)  # libsumo: ensure list type
                    best_length = length

            except Exception:
                continue

        if not best_edges:
            return False

        try:
            route_id = f"route_{random.randint(0, 999_999)}"
            traci.route.add(route_id, best_edges)  # libsumo: must be list, not tuple
            self.current_route_edges = list(best_edges)

            traci.vehicle.add(self.VEH_ID, route_id, departPos="free", typeID=self.VTYPE_ID)

            # IMPORTANT: setSpeedMode / setLaneChangeMode / setSpeed must be called
            # AFTER the vehicle is inserted into simulation (i.e., exists in getIDList).
            # Calling them right after add() causes: "Vehicle 'ego_vehicle' is not known".
            for _ in range(20):
                traci.simulationStep()
                if self._veh_exists():
                    traci.vehicle.setSpeedMode(self.VEH_ID, 0)
                    traci.vehicle.setLaneChangeMode(self.VEH_ID, 0)
                    # Initial speed injection to avoid standing still behind traffic jam
                    traci.vehicle.setSpeed(self.VEH_ID, 5.5)  # ~20 km/h
                    return True
        except Exception as e:
            print(f"[WARN] Spawn error: {e}")

        return False

    # =========================================================================
    # STEP
    # =========================================================================
    def step(self, action: np.ndarray):
        traci.switch(self.label)
        self.step_count += 1

        steer_cmd = float(action[0])
        accel_cmd = float(action[1])

        # --- Early exit if vehicle already gone ---
        if not self._veh_exists():
            return (
                np.zeros(25, dtype=np.float32),
                0.0, True, False,
                self._make_info(0.0, 0.0, 0.0, 0.0, 0, "vanished", "vanished"),
            )

        # --- Apply longitudinal control ---
        # FIX: Use setSpeed() instead of slowDown() to prevent jerky movement.
        # slowDown() is a TEMPORARY command that expires after `duration`, causing
        # the vehicle to stop (since setSpeedMode(0) disables all automatic models).
        # setSpeed() sets a PERSISTENT target speed until the next call.
        desired_accel = accel_cmd * (self.MAX_ACCEL if accel_cmd >= 0 else self.MAX_DECEL)
        delta_t = self.SIM_STEPS * traci.simulation.getDeltaT()
        current_speed = traci.vehicle.getSpeed(self.VEH_ID)
        target_speed = np.clip(current_speed + desired_accel * delta_t, 0.0, self.MAX_SPEED)
        traci.vehicle.setSpeed(self.VEH_ID, target_speed)

        # --- Apply lateral control ---
        offset = 0
        if steer_cmd < -self.LC_THRESHOLD:
            offset = -1
        elif steer_cmd > self.LC_THRESHOLD:
            offset = 1

        if offset != 0:
            try:
                cur_lane = traci.vehicle.getLaneIndex(self.VEH_ID)
                edge = traci.vehicle.getRoadID(self.VEH_ID)
                n_lanes = traci.edge.getLaneNumber(edge)
                desired = cur_lane + offset
                if 0 <= desired < n_lanes:
                    traci.vehicle.changeLane(self.VEH_ID, desired, 2.0)
            except Exception:
                pass

        # --- Simulation micro-steps ---
        acc_energy = 0.0
        sum_speed = 0.0
        valid = 0
        terminated = False
        terminal_event = None  # Wrappers apply rewards based on this

        for _ in range(self.SIM_STEPS):
            traci.simulationStep()

            # ── 1. Check if vehicle naturally arrived ──────────
            try:
                if self.VEH_ID in list(traci.simulation.getArrivedIDList()):
                    self._success = True
                    terminated = True
                    terminal_event = "success"
                    break
            except Exception:
                pass

            # ── 2. Check teleport list ─────────────────────────
            try:
                teleport_ids = list(traci.simulation.getStartingTeleportIDList())
                if self.VEH_ID in teleport_ids:
                    terminated = True
                    terminal_event = "teleport"
                    break
            except Exception:
                pass

            # ── 3. Check collision ─────────────────────────────
            try:
                collisions = traci.simulation.getCollisions()
                col_ids = []
                for c in collisions:
                    if hasattr(c, 'collider'):
                        col_ids.append(c.collider)
                        col_ids.append(c.victim)
                    else:
                        try:
                            col_ids.append(c[0])
                            col_ids.append(c[1])
                        except (IndexError, TypeError):
                            pass
                if self.VEH_ID in col_ids:
                    terminated = True
                    self._success = False
                    terminal_event = "collision"
                    break
            except Exception:
                pass

            # ── 4. Check existence ─────────────────────────────
            if not self._veh_exists():
                terminated = True
                terminal_event = "success" if self._success else "vanished"
                break

            # ── 5. Fallback success check (if close enough) ────
            if not self._success and self._get_dist_to_destination() < 3.0:
                self._success = True
                terminated = True
                terminal_event = "success"
                break


            # Accumulate telemetry
            v = traci.vehicle.getSpeed(self.VEH_ID)
            sum_speed += v
            valid += 1

            e = traci.vehicle.getElectricityConsumption(self.VEH_ID)
            acc_energy += (e if e is not None and not np.isnan(e) else 0.0)

        # --- Post-loop checks ---
        if self._success and terminal_event is None:
            terminated = True
            terminal_event = "success"

        truncated = self.step_count >= self.max_episode_steps
        obs = self._get_obs()
        avg_speed = sum_speed / max(1, valid)

        # Stuck detection: Agent stands completely still or crawls (<0.5m/s)
        if avg_speed < 0.5:
            self.stuck_time += 1
        else:
            self.stuck_time = 0

        # Adjust threshold: 15 macro-steps = 75 sim steps (~7.5 seconds standing still)
        if self.stuck_time > 15 and not terminated:
            terminated = True
            terminal_event = "stuck"

        # Compute telemetry for wrappers
        wiggle = 0.0
        if self.prev_action is not None:
            wiggle = float(np.mean(np.abs(action - self.prev_action)))
        self.prev_action = action.copy()

        safety = 0.0
        if self._veh_exists():
            leader = traci.vehicle.getLeader(self.VEH_ID, self.MAX_DIST)
            if leader is not None:
                ld = leader[1]
                if ld < self.TARGET_DIST:
                    safety = np.clip(1.0 - ld / self.TARGET_DIST, 0.0, 1.0)

        success_flag = 1 if self._success else 0

        # Determine reason string for logging
        if terminal_event:
            reason = terminal_event
        elif truncated:
            reason = "timeout"
        else:
            reason = "running"

        # Base env ALWAYS returns reward=0.0
        # Terminal and step rewards are computed by stage-specific wrappers
        reward = 0.0

        info = self._make_info(avg_speed, acc_energy, wiggle, safety,
                               success_flag, reason, terminal_event)
        return obs, reward, terminated, truncated, info

    def _make_info(self, speed, energy, wiggle, safety, success, reason,
                   terminal_event=None) -> dict:
        return {
            "real_speed": speed,
            "real_energy": energy,
            "wiggle": wiggle,
            "safety": safety,
            "step_reward": 0.0,
            "is_success": success,
            "success_reason": reason,
            "terminal_event": terminal_event,
        }

    # =========================================================================
    # CLOSE
    # =========================================================================
    def close(self):
        try:
            if self._sim_running:
                traci.switch(self.label)
                traci.close()
                self._sim_running = False
        except Exception:
            self._sim_running = False
