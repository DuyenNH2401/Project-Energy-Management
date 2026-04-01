import os

# Đường dẫn map
SUMOCFG_PATH = os.path.join(os.path.dirname(__file__), "map", "map", "run.sumocfg")

# Action Space Definition
NUM_DISCRETE_ACTIONS = 12

# Action mappings: 3 Lateral x 4 Longitudinal
# Lateral: 0=Change Left, 1=Keep Lane, 2=Change Right
LATERAL_ACTIONS = [-1, 0, 1]  

# Longitudinal (Accels m/s^2): 0=Brake, 1=Coast & Regen, 2=Maintain, 3=Accelerate
# Coast will slowly decelerate and regen power, Maintain keeps current speed.
LONGITUDINAL_ACCELS = [-3.0, -0.5, 0.0, 2.0]  

# Vehicle configs
MAX_SPEED = 30.0  # m/s (approx 108 km/h)
STEP_LENGTH = 0.5 # s (Simulation step time)

# Reward weights (Enhanced based on Stage-3 Wrappers)
W_ENERGY = 1.5           # Trọng số cho Penalty Efficiency (Wh/m)
W_TTC = 2.0              # Trọng số an toàn (Dynamic Headway 1.5s)
W_SPEED = 0.5            # Thưởng Gaussian cho tốc độ gần mục tiêu
W_TOO_SLOW = 1.0         # Phạt nặng chống Speed Collapse
W_LANE_CHANGE = -0.1     # Phạt nhẹ chuyển làn không cần thiết (Negative Reward)
W_SMOOTH = 0.5           # Phạt độ giật (Jerky actions)
W_ALIVE = 0.05           # Thưởng sinh tồn (Survival Bonus)
MIN_DESIRED_SPEED = 5.0  # Vận tốc tối thiểu mong đợi (m/s)
