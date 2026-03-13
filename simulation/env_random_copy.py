# env_random.py (Optimized Version)
import os
import sys
import traci
import time
import random
import gymnasium as gym
import numpy as np
from gymnasium import spaces

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

class SumoEnv(gym.Env):
    def __init__(self, render: bool = True, map_config = ["maps/TestMap/osm.sumocfg"], 
              VTYPE_ID = "custom_passenger_car", TRAFFIC_SCALE = 0.5,
              test_mode: bool = False, test_route = "TestMap/test_route.rou.xml",
              imperfection = 0.5, impatience = 0.5, delay = 0) -> None:
        super().__init__()

        self.VEH_ID = "my_ego_car"
        self.VTYPE_ID = VTYPE_ID
        self.TRAFFIC_SCALE = TRAFFIC_SCALE
        self.render_mode: bool = render
        self.test_mode = test_mode
        self.test_route = test_route
        self.delay = delay
        self.step_count = 0
        self.MAX_EPISODE_STEPS = 1001
        
        # --- CONSTANTS ---
        self.MAX_SPEED = 55.6
        self.MAX_ACCEL = 4.15
        self.MAX_DECEL = 6.0
        self.MAX_ELEC = 120
        self.MAX_SLOPE = 20
        self.MAX_DIST = 100
        self.TARGET_DIST = 35.0

        self.maps = [map_config] if isinstance(map_config, str) else map_config
        self.imperfection = imperfection
        self.impatience = impatience
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32)
        
        self.last_known_dist = 0.0
        
        # --- CACHE STORAGE (Nơi lưu dữ liệu tạm để tránh gọi API nhiều lần) ---
        self.veh_data = {} 

    def _veh_exists(self):
        try:
            return self.VEH_ID in traci.vehicle.getIDList()
        except Exception:
            return False

    # --- HÀM MỚI: CẬP NHẬT CACHE ---
    def _update_cache(self):
        """Gọi API Traci 1 lần và lưu tất cả thông tin cần thiết vào biến self.veh_data"""
        if not self._veh_exists():
            self.veh_data = None
            return

        try:
            # Gom nhóm các lệnh gọi API để tiết kiệm thời gian (nếu thư viện hỗ trợ, ở đây ta gọi lần lượt nhưng lưu lại)
            self.veh_data = {
                "speed": traci.vehicle.getSpeed(self.VEH_ID),
                "accel": traci.vehicle.getAcceleration(self.VEH_ID),
                "elec": traci.vehicle.getElectricityConsumption(self.VEH_ID),
                "lane_idx": traci.vehicle.getLaneIndex(self.VEH_ID),
                "lane_id": traci.vehicle.getLaneID(self.VEH_ID),
                "road_id": traci.vehicle.getRoadID(self.VEH_ID),
                "slope": traci.vehicle.getSlope(self.VEH_ID),
                "lat_offset": traci.vehicle.getLateralLanePosition(self.VEH_ID),
                "lane_pos": traci.vehicle.getLanePosition(self.VEH_ID),
                "leader": traci.vehicle.getLeader(self.VEH_ID, dist=self.MAX_DIST),
                "tls": traci.vehicle.getNextTLS(self.VEH_ID)
            }
        except Exception:
            self.veh_data = None

    def _get_passenger_edges(self) -> list:
        all_edges = traci.edge.getIDList()
        valid_edges = []
        for edge_id in all_edges:
            if edge_id.startswith(":") or edge_id.startswith("!"):
                continue
            lane_id = f"{edge_id}_0" 
            try:
                allowed = traci.lane.getAllowed(lane_id)
                if not allowed or "passenger" in allowed:
                    valid_edges.append(edge_id)
            except:
                continue
        return valid_edges
    
    # ------------------------------------------------------------------ #
    #  ROUTE-AWARE LANE GUIDANCE                                          #
    # ------------------------------------------------------------------ #
    def _get_turn_direction_numeric(self, from_edge: str, to_edge: str) -> float:
        """
        Tính hướng rẽ khi chuyển từ from_edge → to_edge.
        Trả về [-1, +1]:  -1 = rẽ trái/U-turn  |  0 = thẳng  |  +1 = rẽ phải
        """
        import math
        try:
            def edge_heading(edge_id):
                shape = traci.edge.getShape(edge_id)
                if len(shape) < 2: return None
                x1, y1 = shape[0]; x2, y2 = shape[-1]
                return math.degrees(math.atan2(y2 - y1, x2 - x1))

            a1 = edge_heading(from_edge)
            a2 = edge_heading(to_edge)
            if a1 is None or a2 is None: return 0.0
            diff = (a2 - a1 + 360) % 360
            if diff > 180: diff -= 360        # [-180, 180)
            # diff > 0 → trái trong hệ SUMO; đổi dấu để +1=phải, -1=trái
            return float(np.clip(-diff / 180.0, -1.0, 1.0))
        except Exception:
            return 0.0

    def _get_next_turn_info(self) -> tuple:
        """
        Nhìn trước trong current_route_edges để tìm khúc rẽ tiếp theo.
        Trả về (turn_dir, turn_dist_norm, lane_offset):
          turn_dir      : hướng rẽ [-1=trái … +1=phải, 0=thẳng]
          turn_dist_norm: tỉ lệ còn lại trên edge hiện tại [1=mới vào/xa rẽ, 0=sắp rẽ]
          lane_offset   : số làn cần dịch [-1=trái … +1=phải], 0 = đang đúng làn rồi
        """
        default = (0.0, 1.0, 0.0)
        if not self.veh_data: return default
        try:
            current_edge = self.veh_data["road_id"]
            if current_edge.startswith(":"): return default
            if not hasattr(self, "current_route_edges") or not self.current_route_edges: return default
            if current_edge not in self.current_route_edges: return default

            indices  = [i for i, x in enumerate(self.current_route_edges) if x == current_edge]
            curr_idx = indices[-1]
            if curr_idx >= len(self.current_route_edges) - 1: return default

            next_edge = self.current_route_edges[curr_idx + 1]
            turn_dir  = self._get_turn_direction_numeric(current_edge, next_edge)

            # Chuẩn hoá theo độ dài TOÀN BỘ edge (không cố định 100m)
            # → xe biết cần chuyển làn sớm dù edge dài hàng km
            lane_id   = self.veh_data["lane_id"]
            if not lane_id: return (turn_dir, 1.0, 0.0)
            lane_len  = traci.lane.getLength(lane_id)
            dist_left = max(0.0, lane_len - self.veh_data["lane_pos"])
            turn_dist_norm = float(np.clip(dist_left / max(lane_len, 1.0), 0.0, 1.0))

            # Tìm làn nào trên current_edge kết nối sang next_edge
            num_lanes    = traci.edge.getLaneNumber(current_edge)
            correct_lane = None
            for li in range(num_lanes):
                try:
                    for link in traci.lane.getLinks(f"{current_edge}_{li}"):
                        if traci.lane.getEdgeID(link[0]) == next_edge:
                            correct_lane = li
                            break
                except Exception:
                    continue
                if correct_lane is not None: break

            if correct_lane is None: return (turn_dir, turn_dist_norm, 0.0)

            # Dương = cần sang phải, âm = cần sang trái, 0 = đúng làn rồi
            raw_offset  = correct_lane - self.veh_data["lane_idx"]
            lane_offset = float(np.clip(raw_offset / max(1, num_lanes - 1), -1.0, 1.0))
            return (turn_dir, turn_dist_norm, lane_offset)
        except Exception:
            return default

    def _get_surroundings(self):
        # Lưu ý: getNeighbors vẫn phải gọi API riêng vì nó trả về danh sách phức tạp
        # Nhưng ta tận dụng self.veh_data["speed"] thay vì gọi lại getSpeed cho xe mình
        
        data = {
            "L_F_Dist": 1.0, "L_F_RelSpeed": 0.0,
            "L_B_Dist": 1.0, "L_B_RelSpeed": 0.0,
            "R_F_Dist": 1.0, "R_F_RelSpeed": 0.0,
            "R_B_Dist": 1.0, "R_B_RelSpeed": 0.0,
        }
        
        # Dùng Cache
        my_speed = self.veh_data["speed"] if self.veh_data else 0.0

        try:
            left_cars = traci.vehicle.getNeighbors(self.VEH_ID, 2)
            closest_front = float("inf")
            closest_back = float("inf")

            for n_id, dist in left_cars:
                if dist > 0: 
                    if dist < closest_front:
                        closest_front = dist
                        n_speed = traci.vehicle.getSpeed(n_id)
                        data["L_F_Dist"] = min(dist, self.MAX_DIST) / self.MAX_DIST
                        data["L_F_RelSpeed"] = (my_speed - n_speed) / self.MAX_SPEED
                else: 
                    if abs(dist) < closest_back:
                        closest_back = abs(dist)
                        n_speed = traci.vehicle.getSpeed(n_id)
                        data["L_B_Dist"] = min(dist, self.MAX_DIST) / self.MAX_DIST
                        data["L_B_RelSpeed"] = (my_speed - n_speed) / self.MAX_SPEED

            right_cars = traci.vehicle.getNeighbors(self.VEH_ID, 1)
            closest_front = float("inf")
            closest_back = float("inf")

            for n_id, dist in right_cars:
                if dist > 0: 
                    if dist < closest_front:
                        closest_front = dist
                        n_speed = traci.vehicle.getSpeed(n_id)
                        data["R_F_Dist"] = min(dist, self.MAX_DIST) / self.MAX_DIST
                        data["R_F_RelSpeed"] = (my_speed - n_speed) / self.MAX_SPEED
                else: 
                    if abs(dist) < closest_back:
                        closest_back = abs(dist)
                        n_speed = traci.vehicle.getSpeed(n_id)
                        data["R_B_Dist"] = min(dist, self.MAX_DIST) / self.MAX_DIST
                        data["R_B_RelSpeed"] = (my_speed - n_speed) / self.MAX_SPEED
        except:
            pass
        
        output = [stats for key, stats in data.items()]
        return output

    def _get_obs(self):
        # Kiểm tra cache
        if self.veh_data is None: 
            return np.zeros(22, dtype=np.float32)

        d = self.veh_data 

        try:
            # 1. EGO PHYSICS
            # Dùng np.clip để chặn giá trị, tránh số quá lớn gây overflow
            velocity = np.clip(d["speed"] / self.MAX_SPEED, 0.0, 2.0)
            acceleration = np.clip(d["accel"] / self.MAX_ACCEL, -1.0, 1.0)
            elec = np.clip(d["elec"] / self.MAX_ELEC, 0.0, 5.0) # Có thể cao hơn 1 chút nếu regeneration
            
            lane_idx = d["lane_idx"]
            road_id = d["road_id"]
            total_lanes = traci.edge.getLaneNumber(road_id)
            norm_lane = lane_idx / max(1, total_lanes - 1)

            slope = np.clip(d["slope"] / self.MAX_SLOPE, -1.0, 1.0)
            lat_offset = np.clip(d["lat_offset"], -10.0, 10.0) # Chặn offset quá lớn

            # 2. SURROUNDINGS
            surroundings = self._get_surroundings()

            # 3. LEADER
            leader = d["leader"]
            if leader:
                l_dist = min(leader[1], self.MAX_DIST) / self.MAX_DIST
                l_rel_speed = (d["speed"] - traci.vehicle.getSpeed(leader[0])) / self.MAX_SPEED
            else:
                l_dist, l_rel_speed = 1.0, 0.0

            # 4. INFRASTRUCTURE
            lane_id = d["lane_id"]
            speed_limit = traci.lane.getMaxSpeed(lane_id) / self.MAX_SPEED if lane_id else 1.0

            tls_data = d["tls"]
            if tls_data:
                tls_dist = min(tls_data[0][2], self.MAX_DIST) / self.MAX_DIST
                tls_state = 1.0 if tls_data[0][3].lower() == 'g' else 0.0
            else:
                tls_dist, tls_state = 1.0, 1.0

            # 5. ROUTE-AWARE LANE GUIDANCE (thay can_left / can_right / turn_dist)
            #   turn_dir    : hướng rẽ tiếp theo  (-1=trái … +1=phải, 0=thẳng)
            #   turn_dist_n : tỉ lệ còn lại trên edge  (1=xa rẽ, 0=sắp rẽ)
            #   lane_offset : số làn cần dịch để vào đúng vị trí  (âm=trái, dương=phải, 0=đúng)
            turn_dir, turn_dist_n, lane_offset = self._get_next_turn_info()

            # Tạo list trước
            obs_list = [
                velocity, acceleration, elec, norm_lane, slope, lat_offset,
                l_dist, l_rel_speed,
                speed_limit, turn_dir, turn_dist_n, tls_dist, tls_state, lane_offset
            ] + surroundings

            # --- SỬA LỖI OVERFLOW Ở ĐÂY ---
            # 1. Chuyển sang float64 trước (để chứa được số lớn nếu có)
            obs = np.array(obs_list, dtype=np.float64)
            
            # 2. Xử lý NaN và Infinity triệt để
            # posinf=1.0, neginf=-1.0 sẽ thay thế số vô cực bằng 1 hoặc -1
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 3. Kẹp giá trị trong khoảng an toàn [-5, 5] để mạng không bị sốc
            obs = np.clip(obs, -5.0, 5.0)

            # 4. Cuối cùng mới ép về float32
            return obs.astype(np.float32)

        except Exception:
            # Nếu có bất kỳ lỗi nào, trả về mảng 0 an toàn
            return np.zeros(22, dtype=np.float32)
    
    def _get_dist_to_destination(self):
        # Hàm này vẫn cần gọi API vì logic phức tạp và không thay đổi thường xuyên trong 1 step
        # Nhưng ta có thể dùng cache["road_id"] và cache["lane_pos"] để tối ưu 1 phần
        try:
            if not self.veh_data: return 1100.0
            current_edge = self.veh_data["road_id"] # Dùng Cache
            
            if current_edge.startswith(":"):
                return self.last_known_dist if self.last_known_dist else 1100.0

            if hasattr(self, "current_route_edges") and current_edge in self.current_route_edges:
                indices = [i for i, x in enumerate(self.current_route_edges) if x == current_edge]
                idx = indices[-1] 
                remaining_edges = self.current_route_edges[idx:]
                dist = 0.0
                for e in remaining_edges:
                    dist += traci.lane.getLength(f"{e}_0")

                dist -= self.veh_data["lane_pos"] # Dùng Cache
                self.last_known_dist = dist
                return dist
            return 1100.0
        except:
            return 1100.0

    def _calculate_reward(self, action):
        if not self.veh_data: return 0.0
        d = self.veh_data

        W_SPEED    =  1.2
        W_PROGRESS =  0.8
        W_ENERGY   = -0.05
        W_COMFORT  = -0.05   # chỉ jerk ga/phanh — KHÔNG phạt tay lái để xe dám chuyển làn
        W_SAFETY   = -0.8
        W_TIME     = -0.2
        W_LANE     = -0.5    # phạt sai làn, tăng dần khi gần khúc rẽ
        W_LANE_OK  =  0.15   # thưởng khi đang đúng làn

        # --- Progress ---
        dist = self._get_dist_to_destination()
        if not hasattr(self, "prev_dist"): self.prev_dist = dist
        progress_reward = np.clip(self.prev_dist - dist, -1.0, 1.0)
        self.prev_dist = dist

        # --- Speed ---
        cur_speed = d["speed"]
        speed_reward = cur_speed / self.MAX_SPEED
        if cur_speed < 3.0:
            speed_reward -= 0.8

        # --- Energy ---
        energy_penalty = np.clip(d["elec"] / self.MAX_ELEC, 0.0, 1.0)

        # --- Comfort: CHỈ jerk của ga/phanh (action[1]), KHÔNG tính tay lái (action[0]) ---
        # Phạt tay lái sẽ làm xe sợ chuyển làn → bỏ hoàn toàn
        if not hasattr(self, "prev_action"): self.prev_action = action
        accel_jerk = float(np.abs(action[1] - self.prev_action[1]))
        self.prev_action = action

        # --- Safety ---
        safety_penalty = 0.0
        leader = d["leader"]
        if cur_speed <= 10.0: target_dist = 15.0
        elif cur_speed <= 20.0: target_dist = 30.0
        else: target_dist = 50.0
        if leader is not None and leader[1] < target_dist:
            safety_penalty = float(np.exp(-(leader[1] / target_dist)))

        # --- Lane guidance ---
        # turn_dist_norm: 1.0 = mới vào edge (xa rẽ), 0.0 = sát cuối edge (sắp rẽ)
        # urgency: 0 khi còn xa, tăng lên 1 khi gần đến cuối edge
        #
        # Nếu đúng làn (lane_offset ≈ 0)  → thưởng W_LANE_OK mỗi bước
        # Nếu sai làn                      → phạt = W_LANE × |offset| × (0.3 + 0.7 × urgency)
        #   hệ số 0.3: phạt nhẹ liên tục dù còn xa rẽ (xe không ngủ quên)
        #   hệ số 0.7: phần tăng thêm khi gần rẽ (buộc xe phải vào đúng làn kịp thời)
        _, turn_dist_n, lane_offset = self._get_next_turn_info()
        urgency    = float(np.clip(1.0 - turn_dist_n, 0.0, 1.0))
        abs_offset = float(np.abs(lane_offset))

        if abs_offset < 0.01:
            lane_reward = W_LANE_OK                                        # đúng làn → thưởng
        else:
            lane_reward = W_LANE * abs_offset * (0.3 + 0.7 * urgency)     # sai làn → phạt

        # --- Tổng hợp ---
        speed_reward    = np.nan_to_num(speed_reward)
        progress_reward = np.nan_to_num(progress_reward)
        energy_penalty  = np.nan_to_num(energy_penalty)
        accel_jerk      = np.nan_to_num(accel_jerk)
        safety_penalty  = np.nan_to_num(safety_penalty)
        lane_reward     = np.nan_to_num(lane_reward)

        reward = (speed_reward    * W_SPEED)    + \
                 (progress_reward * W_PROGRESS) + \
                 (accel_jerk      * W_COMFORT)  + \
                 (safety_penalty  * W_SAFETY)   + \
                 (energy_penalty  * W_ENERGY)   + \
                 lane_reward                    + \
                 W_TIME
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        
        try:
            # Tối ưu: Dùng traci.load() thay vì close/start nếu có thể
            # (Giữ nguyên logic cũ nếu bạn muốn an toàn, hoặc thay đổi nếu muốn nhanh hơn nữa)
            traci.close()
        except Exception:
            pass
        time.sleep(0.5) # Giảm delay xuống chút
        
        if self.test_mode:
            active_map = self.maps[0]
            route_arg = ["-a", self.test_route]
        else:
            active_map = random.choice(self.maps)
            route_arg = []
            
        self.step_count = 0
        self.veh_data = None # Reset cache
        
        SumoBinary = "sumo-gui" if self.render_mode else "sumo"
        SumoCMD = [SumoBinary, "-c", active_map] + route_arg + \
                ["--start", "--quit-on-end",
                "--device.emissions.probability", "1.0",
                "--scale", str(self.TRAFFIC_SCALE),
                "--delay", str(self.delay),
                "--no-step-log", "true",
                "--time-to-teleport", "-1",
                "--collision.action", "remove",
                "--collision.check-junctions", "true",
                "--no-warnings", "true"]
        
        traci.start(SumoCMD)
        self._success = False

        if hasattr(self, "prev_action"): del self.prev_action
        if hasattr(self, "prev_dist"): del self.prev_dist
        
        # Warmup
        for _ in range(random.randint(300, 600)): traci.simulationStep()

        try:
            existing_types = traci.vehicletype.getIDList()
            source_type = "DEFAULT_VEHTYPE"
            if source_type not in existing_types and len(existing_types) > 0:
                source_type = existing_types[0]

            traci.vehicletype.copy(source_type, self.VTYPE_ID)
            traci.vehicletype.setVehicleClass(self.VTYPE_ID, "passenger")
            traci.vehicletype.setColor(self.VTYPE_ID, (0, 255, 0)) 
            traci.vehicletype.setParameter(self.VTYPE_ID, "mass", "2911")
            traci.vehicletype.setLength(self.VTYPE_ID, "5.1181")
            traci.vehicletype.setEmissionClass(self.VTYPE_ID, "MMPEVEM")
            
            traci.vehicletype.setParameter(self.VTYPE_ID, "has.battery.device", "true")
            traci.vehicletype.setParameter(self.VTYPE_ID, "device.battery.capacity", "123000.00")
            traci.vehicletype.setParameter(self.VTYPE_ID, "device.battery.chargeLevel", "123000.00")
            traci.vehicletype.setParameter(self.VTYPE_ID, "device.battery.rechargeEfficiency", "0.8")
            traci.vehicletype.setParameter(self.VTYPE_ID, "device.battery.maxRegenerationAcceleration", "2.0")
            
            for v_type in existing_types:
                traci.vehicletype.setParameter(v_type, "sigma", str(self.imperfection))
                traci.vehicletype.setParameter(v_type, "impatience", str(self.impatience))
        except Exception:
            pass

        spawned = False
        if self.test_mode:
            while self.VEH_ID not in traci.vehicle.getIDList():
                traci.simulationStep()
            traci.vehicle.setType(self.VEH_ID, self.VTYPE_ID)
            traci.vehicle.setSpeedMode(self.VEH_ID, 0)
            traci.vehicle.setLaneChangeMode(self.VEH_ID, 0)
            for _ in range(50):
                traci.simulationStep()
                if self.VEH_ID in traci.vehicle.getIDList():
                    spawned = True
                    break
        else:
            # FIX A: Luôn làm mới drivable_edges sau mỗi traci.start().
            # Nếu cache thì edge IDs của map cũ sẽ sai khi map xoay vòng.
            self.drivable_edges = self._get_passenger_edges()

            for attempt in range(20):
                if not self.drivable_edges: break
                start_edge = random.choice(self.drivable_edges)
                route_edges = [start_edge]
                # FIX B: Set visited tránh mọi cycle, không chỉ quay bước ngay trước.
                visited_edges = {start_edge}
                current_len = 0.0
                try:
                    current_len += traci.lane.getLength(f"{start_edge}_0")
                except:
                    continue
                
                curr_edge_id = start_edge
                dead_end = False
                while current_len < 1100.0:
                    # FIX C: Query TẤT CẢ các làn của edge hiện tại.
                    # Chỉ query _0 (làn phải nhất) khiến rẽ trái/U-turn bị bỏ qua hoàn toàn
                    # vì các nhánh đó chỉ xuất phát từ làn trong (_1, _2...).
                    try:
                        num_lanes = traci.edge.getLaneNumber(curr_edge_id)
                    except:
                        dead_end = True
                        break
                    valid_next_edges = []
                    for lane_idx in range(num_lanes):
                        try:
                            links = traci.lane.getLinks(f"{curr_edge_id}_{lane_idx}")
                        except:
                            continue
                        for link in links:
                            next_lane_id = link[0]
                            try:
                                next_edge_id = traci.lane.getEdgeID(next_lane_id)
                            except:
                                continue
                            if (not next_edge_id.startswith(":")
                                    and next_edge_id in self.drivable_edges
                                    and next_edge_id not in visited_edges
                                    and next_edge_id not in valid_next_edges):
                                valid_next_edges.append(next_edge_id)
                    if not valid_next_edges:
                        dead_end = True
                        break
                    next_edge = random.choice(valid_next_edges)
                    route_edges.append(next_edge)
                    visited_edges.add(next_edge)
                    try: current_len += traci.lane.getLength(f"{next_edge}_0")
                    except: pass
                    curr_edge_id = next_edge
                
                if not dead_end and current_len >= 1100.0:
                    try:
                        route_id = f"route_{random.randint(0, 999999)}"
                        traci.route.add(route_id, route_edges)
                        self.current_route_edges = route_edges
                        traci.vehicle.add(self.VEH_ID, route_id, departPos="free", typeID=self.VTYPE_ID)
                        for _ in range(50):
                            traci.simulationStep()
                            if self.VEH_ID in traci.vehicle.getIDList():
                                spawned = True
                                break
                        if spawned:
                            traci.vehicle.setSpeedMode(self.VEH_ID, 0)
                            traci.vehicle.setLaneChangeMode(self.VEH_ID, 0)
                            self.last_known_dist = current_len
                            self.prev_dist = current_len
                            break 
                        else:
                            try: traci.vehicle.remove(self.VEH_ID)
                            except: pass
                    except Exception:
                        try: traci.vehicle.remove(self.VEH_ID)
                        except: pass
                        continue
            
            if not spawned:
                return self.reset(seed=seed, options=options)

        if self.render_mode and spawned:
            if self.VEH_ID in traci.vehicle.getIDList():
                traci.gui.trackVehicle("View #0", self.VEH_ID)
                traci.gui.setZoom("View #0", 1001)

        self.stuck_time = 0
        # --- CẬP NHẬT CACHE LẦN ĐẦU ---
        self._update_cache() 
        
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.step_count += 1
        steer_cmd = action[0]
        accel_cmd = action[1]
        
        if accel_cmd >= 0:
            desired_accel = accel_cmd * self.MAX_ACCEL
        else:
            desired_accel = accel_cmd * self.MAX_DECEL 

        SIM_STEPS = 1 
        
        if not self._veh_exists():
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, \
                {"real_speed": 0, "reason": "already_dead", "is_success": 0}

        # Áp dụng action (không cần cache vì là lệnh gửi đi)
        traci.vehicle.setAcceleration(self.VEH_ID, desired_accel, duration=0.5)
        LC_THRESHOLD = 0.3
        current_lane_idx = traci.vehicle.getLaneIndex(self.VEH_ID)
        if steer_cmd < -LC_THRESHOLD: 
            traci.vehicle.changeLane(self.VEH_ID, max(0, current_lane_idx - 1), 1.0)
        elif steer_cmd > LC_THRESHOLD: 
            try:
                edge_id = traci.vehicle.getRoadID(self.VEH_ID)
                num_lanes = traci.edge.getLaneNumber(edge_id)
                traci.vehicle.changeLane(self.VEH_ID, min(num_lanes - 1, current_lane_idx + 1), 1.0)
            except: pass

        reward = 0.0
        terminated = False
        truncated = False
        accumulated_energy = 0.0 
        sum_speed = 0.0
        valid_steps = 0 
        termination_reason = "running"

        for _ in range(SIM_STEPS):
            traci.simulationStep()

            # --- QUAN TRỌNG: CẬP NHẬT CACHE NGAY SAU KHI MÔ PHỎNG ---
            self._update_cache()
            
            # Nếu xe chết thì không có cache, break
            if self.veh_data is None:
                terminated = True
                teleport_list = traci.simulation.getStartingTeleportIDList()
                if self.VEH_ID in teleport_list:
                    reward -= 50.0 
                    termination_reason = "teleport"
                else:
                    reward -= 200.0 
                    termination_reason = "collision"
                break 
            
            # --- DÙNG DỮ LIỆU TỪ CACHE (self.veh_data) ---
            ego_speed = self.veh_data["speed"]
            leader = self.veh_data["leader"]
            
            # Context Aware Stuck Detection
            # Check Đèn Đỏ (Dùng Cache)
            is_red_light = False
            tls_data = self.veh_data["tls"]
            if tls_data and tls_data[0][2] < 20.0:
                state = tls_data[0][3].lower()
                if 'r' in state or 'y' in state: is_red_light = True
            
            # Check Xe Trước Dừng
            is_leader_stopped = (leader is not None and traci.vehicle.getSpeed(leader[0]) < 0.5)

            if ego_speed < 0.5 and not (is_red_light or is_leader_stopped):
                reward -= 0.5
                self.stuck_time += 1
            else:
                self.stuck_time = 0

            sum_speed += ego_speed
            valid_steps += 1
            # Dùng cache
            e = self.veh_data["elec"]
            accumulated_energy += e if not np.isnan(e) else 0.0

            # Tính reward (Hàm này đã được sửa để dùng self.veh_data)
            reward += self._calculate_reward(action) / SIM_STEPS

            if self._success_check():
                terminated = True
                reward += 200.0
                self._success = True
                termination_reason = "goal"
                break
        
        if not hasattr(self, "prev_action"): self.prev_action = action
        action_delta = np.abs(action - self.prev_action)
        # Chỉ log jerk ga/phanh (khớp với reward) — không đo tay lái
        wiggle_stat = float(action_delta[1])
        self.prev_action = action

        if not terminated:
            if self.stuck_time > 100:
                terminated = True
                reward -= 400.0
                termination_reason = "stuck_too_long"
            elif self.step_count >= self.MAX_EPISODE_STEPS:
                truncated = True
                termination_reason = "timeout"

        # _get_obs sẽ tự dùng self.veh_data hiện tại (đã update trong loop)
        obs = self._get_obs()
        avg_real_speed = sum_speed / max(1, valid_steps)

        safety_val = 0.0
        if self.veh_data and self.veh_data["leader"]:
            dist = self.veh_data["leader"][1]
            if dist < 20: safety_val = 1.0 - (dist/20.0) 

        info = {
            "real_speed": avg_real_speed,
            "real_energy": accumulated_energy, 
            "wiggle": wiggle_stat,
            "safety": safety_val, 
            "step_reward": reward,
            "is_success": 1 if self._success else 0,
            "reason": termination_reason 
        }

        return obs, reward, terminated, truncated, info

    def _success_check(self):
        if not self.veh_data: return False
        try:
            current_edge = self.veh_data["road_id"]
            if current_edge.startswith(":"): return False 

            if hasattr(self, "current_route_edges") and self.current_route_edges:
                if current_edge == self.current_route_edges[-1]:
                    # Vẫn cần gọi API lấy length vì cái này static
                    lane_len = traci.lane.getLength(self.veh_data["lane_id"])
                    pos = self.veh_data["lane_pos"]
                    if pos > (lane_len - 20.0):
                        return True
        except: pass
        return False

    def close(self):
        try:
            traci.close()
        except:
            pass

if __name__ == "__main__":
    env = SumoEnv(map_config="TestMap/osm.sumocfg", render=True, test_mode=False, test_route="TestMap/test_route.rou.xml", delay=100)
    obs, info = env.reset()
    print(f"Init Obs Shape: {obs.shape}")
    done = False
    total_reward = 0
    print("Starting Loop...")
    for i in range(50):
        action = env.action_space.sample()
        action[1] = np.random.uniform(-1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {env.step_count} | Action: {action} | Reward: {reward:.2f} | Speed: {obs[0]:.2f} | Energy: {obs[2]:.2f}")
        if terminated or truncated:
            print("Episode Finished!")
            obs, info = env.reset()
    env.close()
    print("Test Complete.")