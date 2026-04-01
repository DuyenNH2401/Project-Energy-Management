import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import libtraci as traci

from config import SUMOCFG_PATH, NUM_DISCRETE_ACTIONS, LATERAL_ACTIONS, LONGITUDINAL_ACCELS, MAX_SPEED, STEP_LENGTH
from config import W_ENERGY, W_TTC, W_SPEED, W_LANE_CHANGE, W_TOO_SLOW, W_SMOOTH, W_ALIVE, MIN_DESIRED_SPEED

class SumoDiscreteEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, use_gui=False):
        super(SumoDiscreteEnv, self).__init__()
        self.use_gui = use_gui
        self.sumo_cmd = ['sumo-gui' if use_gui else 'sumo', '-c', SUMOCFG_PATH, '--step-length', str(STEP_LENGTH)]
        
        if self.use_gui:
            self.sumo_cmd.extend(['--start', '--delay', '150'])
            
        # Action Space: 12 Discrete Actions (3x4)
        self.action_space = spaces.Discrete(NUM_DISCRETE_ACTIONS)
        
        # State Space: 25-dim vector for Hybrid Mamba
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32)
        
        self.ego_id = "ego_vehicle"
        self.vtype_id = "vinfast_vf9"
        self.steps = 0
        self.max_steps = 1000

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        try:
            traci.close()
        except Exception:
            pass
            
        traci.start(self.sumo_cmd)
        
        self._setup_vf9()
        
        # Sinh route ngẫu nhiên an toàn
        routes = [r for r in traci.route.getIDList() if not r.startswith("!")]
        if len(routes) > 0:
            route_id = np.random.choice(routes)
            traci.vehicle.add(self.ego_id, route_id, typeID=self.vtype_id)
        else:
            edges = traci.edge.getIDList()
            valid_edges = [e for e in edges if not e.startswith(":") and traci.edge.getLaneNumber(e) >= 1]
            for i in range(50):
                start_edge, end_edge = np.random.choice(valid_edges, 2)
                route = traci.simulation.findRoute(start_edge, end_edge)
                if route.edges: # Route is found
                    route_id = f"ego_route_{i}"
                    traci.route.add(route_id, route.edges)
                    traci.vehicle.add(self.ego_id, route_id, typeID=self.vtype_id)
                    break
            
        if self.ego_id in traci.vehicle.getIDList():
            traci.vehicle.setSpeedMode(self.ego_id, 0) # Tắt tự động điều chỉnh tốc độ của SUMO
            traci.vehicle.setLaneChangeMode(self.ego_id, 0) # Tắt tự động nhảy làn
            
        self.steps = 0
        self.prev_lat_idx = 1
        self.prev_long_idx = 2
        # Bước một dummy step để có ego vehicle trên map
        traci.simulationStep()
        
        if self.use_gui and self.ego_id in traci.vehicle.getIDList():
            try:
                traci.gui.trackVehicle("View #0", self.ego_id)
                traci.gui.setZoom("View #0", 1500)
            except Exception:
                pass
                
        obs = self._get_obs()
        return obs, {}

    def _setup_vf9(self):
        try:
            existing = traci.vehicletype.getIDList()
            source = "DEFAULT_VEHTYPE" if "DEFAULT_VEHTYPE" in existing else (existing[0] if existing else "")
            
            if source:
                traci.vehicletype.copy(source, self.vtype_id)
                
            traci.vehicletype.setVehicleClass(self.vtype_id, "passenger")
            traci.vehicletype.setColor(self.vtype_id, (0, 255, 0)) # Xe VF9 sẽ hiển thị màu xanh lá rực
            traci.vehicletype.setParameter(self.vtype_id, "mass", "2911")
            traci.vehicletype.setLength(self.vtype_id, 5.1181)
            traci.vehicletype.setEmissionClass(self.vtype_id, "MMPEVEM")
            traci.vehicletype.setParameter(self.vtype_id, "has.battery.device", "true")
            traci.vehicletype.setParameter(self.vtype_id, "device.battery.capacity", "123000.00")
            traci.vehicletype.setParameter(self.vtype_id, "device.battery.chargeLevel", "123000.00")
        except Exception as e:
            print(f"[WARN] VF9 setup error: {e}")

    def step(self, action):
        done = False
        truncated = False
        reward = 0.0
        
        # Giải mã action
        lat_idx = action // 4 # 0=Left, 1=Keep, 2=Right
        long_idx = action % 4 # 0=Brake, 1=Coast, 2=Maintain, 3=Accel
        
        lat_val = LATERAL_ACTIONS[lat_idx]
        long_val = LONGITUDINAL_ACCELS[long_idx]
        
        ego_active = self.ego_id in traci.vehicle.getIDList()
        curr_speed = 0.0
        
        if ego_active:
            curr_speed = traci.vehicle.getSpeed(self.ego_id)
            curr_lane = traci.vehicle.getLaneIndex(self.ego_id)
            edge_id = traci.vehicle.getRoadID(self.ego_id)
            num_lanes = traci.edge.getLaneNumber(edge_id) if not edge_id.startswith(":") else 1
            
            # Action: Longitudinal
            new_speed = max(0.0, min(MAX_SPEED, curr_speed + long_val * STEP_LENGTH))
            traci.vehicle.setSpeed(self.ego_id, new_speed)
            
            # Action: Lateral
            if lat_val != 0:
                target_lane = curr_lane + lat_val
                if 0 <= target_lane < num_lanes:
                    traci.vehicle.changeLane(self.ego_id, target_lane, duration=STEP_LENGTH)
                    
        # Update simulation round
        traci.simulationStep()
        self.steps += 1
        
        ego_active_after = self.ego_id in traci.vehicle.getIDList()
        if not ego_active_after:
            # Reached destination or teleported
            done = True
            reward += 100.0 # Bouns reaching dest
        elif self.steps >= self.max_steps:
            truncated = True
            
        obs = self._get_obs()
        
        if ego_active and not done:
            reward += self._compute_reward(lat_idx, long_idx, curr_speed)
            
        # Thêm reward phụ thuộc va chạm nếu có
        
        self.prev_lat_idx = lat_idx
        self.prev_long_idx = long_idx
        return obs, reward, done, truncated, {}

    def _get_obs(self):
        obs = np.zeros(25, dtype=np.float32)
        if self.ego_id in traci.vehicle.getIDList():
            speed = traci.vehicle.getSpeed(self.ego_id)
            pos2d = traci.vehicle.getPosition(self.ego_id)
            angle = traci.vehicle.getAngle(self.ego_id)
            lane_pos = traci.vehicle.getLanePosition(self.ego_id)
            
            obs[0] = speed
            obs[1] = angle
            obs[2] = lane_pos
            obs[3] = pos2d[0]
            obs[4] = pos2d[1]
            
            # Get leader
            leader = traci.vehicle.getLeader(self.ego_id, dist=100.0)
            if leader:
                l_id, dist = leader
                obs[5] = dist
                obs[6] = speed - traci.vehicle.getSpeed(l_id)
            else:
                obs[5] = 100.0
                obs[6] = 0.0
                
            # Get follower
            follower = traci.vehicle.getFollower(self.ego_id, dist=100.0)
            if follower and follower[0]:
                f_id, dist = follower
                obs[7] = dist
                obs[8] = traci.vehicle.getSpeed(f_id) - speed
            else:
                obs[7] = 100.0
                obs[8] = 0.0
                
        return obs

    def _compute_reward(self, lat_idx, long_idx, speed):
        reward = 0.0
        
        # 0. R_ALIVE (Survival Bonus)
        reward += W_ALIVE
        
        # 1. R_ENERGY_EFFICIENCY (Wh/m)
        if long_idx == 0: power_w = -80.0
        elif long_idx == 1: power_w = -50.0
        elif long_idx == 2: power_w = 30.0
        elif long_idx == 3: power_w = 150.0
        else: power_w = 0.0
        
        if speed > 0.5:
            energy_rate = power_w / speed
            if energy_rate > 0: # Energy consumed
                r_efficiency = np.clip(energy_rate / 0.5, 0.0, 1.0)
                reward -= W_ENERGY * r_efficiency
            else: # Regenerative braking
                r_regen = np.clip(abs(energy_rate) / 0.5, 0.0, 1.0)
                reward += W_ENERGY * r_regen * 0.5 # giving back half the benefit
        else:
            reward -= W_ENERGY * 1.0 # Max penalty for speed collapse causing infinite Wh/m
        
        # 2. R_SAFETY_LEADER (TTC-based Headway)
        leader = traci.vehicle.getLeader(self.ego_id, 200.0)
        if leader:
            l_id, dist = leader
            safe_dist = speed * 1.5 + 5.0 # 1.5s headway + 5m min gap
            if dist < safe_dist:
                r_leader = (safe_dist - dist) / safe_dist
                reward -= W_TTC * r_leader
                    
        # 3. R_SPEED_TARGET & SPEED COLLAPSE
        target_speed = MAX_SPEED * 0.8
        speed_error = (speed - target_speed) / max(target_speed, 1.0)
        r_speed_target = np.exp(-3.0 * speed_error ** 2)
        reward += W_SPEED * r_speed_target
        
        if speed < MIN_DESIRED_SPEED:
            r_too_slow = 2.0 * (1.0 - (speed / MIN_DESIRED_SPEED))
            reward -= W_TOO_SLOW * r_too_slow
            
        # 4. R_SMOOTH_CONTROL (Jerky action penalties)
        if hasattr(self, 'prev_long_idx') and hasattr(self, 'prev_lat_idx'):
            r_smooth_accel = abs(long_idx - getattr(self, "prev_long_idx", 2))
            r_smooth_steer = abs(lat_idx - getattr(self, "prev_lat_idx", 1))
            # scale actions diff: long range is 0-3, lat range 0-2
            reward -= W_SMOOTH * (r_smooth_accel / 3.0 + r_smooth_steer / 2.0)
        
        # 5. R_LANE_CHANGE (Freq limit)
        if lat_idx != 1:
            reward += W_LANE_CHANGE
            
        return float(reward)

    def close(self):
        try:
            traci.close()
        except Exception:
            pass
