#Điền model và chạy file này để plot ra đồ hình
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_tianshou_metrics(csv_path, output_dir='reports/tianshou_ppo_plot_6_random'):
    # 1. Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Đã tạo thư mục: {output_dir}")

    # 2. Đọc dữ liệu
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return

    # Giả định trục X là số Episode hoặc dòng (Index) nếu không có cột 'step' hay 'time'
    if 'episode' in df.columns:
        x_axis = 'episode'
    elif 'step' in df.columns:
        x_axis = 'step'
    else:
        df['index'] = df.index
        x_axis = 'index'

    # Danh sách các chỉ số cần vẽ (dạng số)
    numerical_metrics = ['ep_reward', 'avg_speed', 'total_energy', 'wiggle', 'safety', 'success']
    
    # 3. Vẽ các chỉ số dạng số
    for metric in numerical_metrics:
        if metric in df.columns:
            plt.figure(figsize=(12, 6))
            # Vẽ đường gốc với độ mờ thấp
            sns.lineplot(data=df, x=x_axis, y=metric, alpha=0.3, label=f'{metric} (raw)')
            # Vẽ đường trung bình trượt (rolling mean) để dễ nhìn xu hướng
            df[f'{metric}_smooth'] = df[metric].rolling(window=min(50, len(df)//10), min_periods=1).mean()
            sns.lineplot(data=df, x=x_axis, y=f'{metric}_smooth', linewidth=2, label=f'{metric} (smooth)')
            
            plt.title(f'Biểu đồ {metric} theo thời gian', fontsize=15)
            plt.xlabel('Thời gian (Episode/Step)', fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            
            # Lưu ảnh
            file_name = f"{metric}_plot.png"
            plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight')
            plt.close()
            print(f"Đã lưu đồ thị: {file_name}")

    # 4. Xử lý riêng cho cột 'reason' (Dữ liệu phân loại)
    if 'reason' in df.columns:
        plt.figure(figsize=(12, 6))
        # Vẽ phân phối của các lý do kết thúc theo thời gian
        # Chia dữ liệu thành các bins (khoảng) để xem sự thay đổi tỉ lệ lý do
        df['bin'] = pd.cut(df.index, bins=10, labels=[f"Part {i+1}" for i in range(10)])
        reason_counts = df.groupby(['bin', 'reason']).size().unstack(fill_value=0)
        
        reason_counts.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='viridis')
        plt.title('Phân bổ lý do kết thúc (Reason) theo các giai đoạn', fontsize=15)
        plt.xlabel('Giai đoạn tập luyện', fontsize=12)
        plt.ylabel('Số lượng', fontsize=12)
        plt.legend(title='Reason', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        file_name = "reason_distribution_plot.png"
        plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight')
        plt.close()
        print(f"Đã lưu đồ thị: {file_name}")


plot_tianshou_metrics('reports/tianshou_ppo/training_log_12032026_232128.csv')