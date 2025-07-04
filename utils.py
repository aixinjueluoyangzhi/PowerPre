from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
import os

def load_and_preprocess(path):
    df = pd.read_csv(path, parse_dates=['Date'])
    df = df.sort_values('Date')

    features = [
        'Global_active_power', 'Voltage', 'Global_intensity',
        'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Sub_metering_4',
        'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU'
    ]
    df = df[features]
    return df

def create_sequences(data, input_len, pred_len, stride=1):
    X, y = [], []
    for i in range(0, len(data) - input_len - pred_len + 1, stride):
        X.append(data[i:i+input_len])
        y.append(data[i+input_len:i+input_len+pred_len, 0])  # 预测目标列
    return np.array(X), np.array(y)

def normalize_data(train_df, test_df=None):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    if test_df is not None:
        test_scaled = scaler.transform(test_df)
        return train_scaled, test_scaled, scaler
    return train_scaled, scaler

# 配置日志
def setup_logging(modle):
    if not os.path.exists('logs'):
        os.makedirs('logs')
    log_filename = f"logs/{modle}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def plot_prediction(pred, true, title, filename):
    plt.figure(figsize=(12, 4))
    plt.plot(true[100], label='True')
    plt.plot(pred[100], label='Predicted')
    plt.title(title)
    plt.legend()
    plt.grid()

    # 确保plots目录存在
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # 保存图片
    plt.savefig(f'plots/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形，避免内存泄漏