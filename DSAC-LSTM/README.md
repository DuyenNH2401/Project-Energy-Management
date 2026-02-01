# DSAC-LSTM: Distributional Soft Actor-Critic with LSTM

Triển khai thuật toán DSAC kết hợp LSTM cho các bài toán Partially Observable (POMDP).

## Cấu trúc Dự án

```
my-code/
├── algorithms/
│   ├── __init__.py
│   └── dsac_lstm.py          # Thuật toán DSAC-LSTM chính
├── buffers/
│   ├── __init__.py
│   └── recurrent_replay_buffer.py  # Episode-based replay buffer
├── networks/
│   ├── __init__.py
│   └── recurrent_networks.py  # LSTM Actor/Critic networks
├── utils/
│   ├── __init__.py
│   └── common.py              # Utilities
├── train.py                   # Training script
├── config.yaml                # Configuration file
└── README.md
```

## Cài đặt

```bash
pip install torch numpy gymnasium tqdm
```

## Sử dụng

### Training cơ bản

```bash
python train.py --env Pendulum-v1 --total_steps 100000
```

### Training với tham số tùy chỉnh

```bash
python train.py \
    --env HalfCheetah-v4 \
    --total_steps 1000000 \
    --batch_size 64 \
    --sequence_length 80 \
    --burn_in_steps 20 \
    --lstm_hidden_dim 256 \
    --hidden_dim 256 \
    --lr 3e-4 \
    --device cuda
```

## Các Tham số Chính

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--env` | Pendulum-v1 | Gymnasium environment ID |
| `--total_steps` | 100000 | Tổng số bước training |
| `--batch_size` | 32 | Kích thước batch |
| `--sequence_length` | 64 | Độ dài chuỗi cho LSTM |
| `--burn_in_steps` | 16 | Số bước burn-in (bỏ qua loss) |
| `--lstm_hidden_dim` | 128 | Kích thước hidden state LSTM |
| `--hidden_dim` | 256 | Kích thước MLP hidden |
| `--gamma` | 0.99 | Discount factor |
| `--tau` | 0.005 | Soft update coefficient |
| `--lr` | 3e-4 | Learning rate |

## Đặc điểm Kỹ thuật

### 1. Input Enrichment
Input vào LSTM bao gồm: `[obs_t, action_{t-1}, reward_{t-1}]` để encode đầy đủ ngữ cảnh.

### 2. Causal Q-Network
Action hiện tại được concatenate **sau** LSTM để đảm bảo tính nhân quả.

### 3. Burn-in Strategy
Bỏ qua `burn_in_steps` bước đầu tiên khi tính loss để ổn định hidden state.

### 4. Distributional Critic
Q-network output (mean, std) thay vì scalar value cho distributional RL.

## Tham khảo

- [DSAC-v2 Paper](https://arxiv.org/abs/2310.05858)
- [R2D2: Recurrent Experience Replay](https://openreview.net/forum?id=r1lyTjAqYX)
