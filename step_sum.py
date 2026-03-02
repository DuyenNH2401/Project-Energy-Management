import pandas as pd

file = "reports/tianshou_ppo/training_log_11022026_001709.csv"

data = pd.read_csv(file)

all_steps = data['steps'].sum()
print(all_steps)