import os
import time
import numpy as np
from stable_baselines3 import SAC, PPO 

import config as cfg
from sumo_env import SumoEnv
from wrappers import FullControlWrapper

# NOTE: Nếu bạn muốn test với model đã train, hãy điền đường dẫn vào đây.
# Ví dụ: MODEL_PATH = "models/baselines/flat_ppo/best/best_model.zip"
# Nếu để None, xe sẽ chạy với hành động ngẫu nhiên (Random actions).
MODEL_PATH = None 

def main():
    print("=" * 60)
    print("  RENDER TEST (5 EPISODES)")
    print("=" * 60)

    # Khởi tạo môi trường với render=True để bật thanh GUI
    # Lưu ý: Cần dùng libtraci hoặc traci để GUI hoạt động trên Windows
    print("Đang khởi tạo môi trường SUMO với GUI...")
    train_env = FullControlWrapper(
        SumoEnv(render=True, traffic_scale=cfg.TRAFFIC_SCALE)
    )

    model = None
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        print(f"Đang tải model đã train từ: {MODEL_PATH}")
        # Lưu ý: Đổi hàm SAC.load thành PPO.load nếu bạn dùng thuật toán PPO
        try:
            model = SAC.load(MODEL_PATH, env=train_env)
            print("Tải model thành công!")
        except Exception as e:
            print(f"Lỗi khi tải model: {e}")
            print("Chuyển sang chạy với Random actions.")
            model = None
    else:
        if MODEL_PATH:
            print(f"Không tìm thấy file: {MODEL_PATH}")
        print("Sử dụng RANDOM actions (Hành động ngẫu nhiên) để test render.")

    num_episodes = 5
    
    for episode in range(1, num_episodes + 1):
        print(f"\n--- Bắt đầu Episode {episode}/{num_episodes} ---")
        obs, info = train_env.reset()
        done = False
        truncated = False
        step = 0
        total_reward = 0.0
        
        while not (done or truncated):
            if model:
                # Dùng model đã train
                action, _states = model.predict(obs, deterministic=True)
            else:
                # Random actions
                action = train_env.action_space.sample()
                
            obs, reward, done, truncated, info = train_env.step(action)
            total_reward += reward
            step += 1
            
            # Bạn có thể điều chỉnh thời gian sleep để làm hình ảnh chậm/nhanh hơn
            time.sleep(0.01)
            
        # Lấy lý do kết thúc episode (collision, success, stuck, timeout, v.v.)
        reason = info.get('terminal_event', 'unknown') if info else 'unknown'
        print(f"Episode {episode} kết thúc | Số bước đi: {step} | Tổng Reward: {total_reward:.2f} | Lý do: {reason}")
        
    print("\nHoàn tất 5 episodes. Đang đóng môi trường...")
    train_env.close()

if __name__ == "__main__":
    main()
