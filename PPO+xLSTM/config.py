# =============================================================================
# Training Configuration for PPO + xLSTM
# =============================================================================

# Environment Settings
ENV_ID = "CartPole-v1"  # Gym environment ID
SEED = 42               # Random seed
DEVICE = "auto"         # Device (cpu/cuda/auto)

# Training Hyperparameters
TOTAL_TIMESTEPS = 100_000   # Tổng số timesteps để train
ROLLOUT_STEPS = 2048        # Số steps thu thập trước mỗi update
PPO_EPOCHS = 4              # Số epoch PPO trên mỗi batch
LR = 3e-4                   # Learning rate
GAMMA = 0.99                # Discount factor
GAE_LAMBDA = 0.95           # GAE lambda
CLIP_EPS = 0.2              # PPO clip epsilon
ENTROPY_COEF = 0.01         # Entropy coefficient
VALUE_LOSS_COEF = 0.5       # Value loss coefficient
MAX_GRAD_NORM = 0.5         # Max gradient norm

# xLSTM Architecture
EMBEDDING_DIM = 64          # xLSTM embedding dimension (số lượng neuron network mỗi lớp)
NUM_BLOCKS = 4              # Số xLSTM blocks (số lượng lớp neuron network)

NUM_HEADS = 4               # Số heads cho sLSTM/mLSTM
# Giữ nguyên hoặc điều chỉnh sao cho EMBEDDING_DIM chia hết cho tham số này

BLOCK_TYPE = "slstm"        # Loại block xLSTM: "slstm", "mlstm", "mixed"

# Logging and Saving
LOG_INTERVAL = 5            # In log mỗi N updates
SAVE_PATH = None            # Đường dẫn lưu model (ví dụ: "models/ppo_xlstm.pt")
