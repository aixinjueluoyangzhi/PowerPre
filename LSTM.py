import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils import load_and_preprocess, normalize_data, create_sequences, setup_logging, plot_prediction
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = setup_logging(modle='LSTM')


class LSTMForecast(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_len):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_len)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out


def train_model(model, train_loader, epochs=50, lr=1e-3):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs+1):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"Epoch {epoch}: Loss={total_loss / len(train_loader):.4f}")


def evaluate(model, X, y, scaler):
    model.eval()
    with torch.no_grad():
        X = torch.tensor(X, dtype=torch.float32).to(device)
        pred = model(X).cpu().numpy()

    mse = mean_squared_error(y, pred)
    mae = mean_absolute_error(y, pred)

    gmin, gmax = scaler.data_min_[0], scaler.data_max_[0]
    pred = pred * (gmax - gmin) + gmin
    y = y * (gmax - gmin) + gmin

    logger.info(f"MSE: {mse:.4f}, MAE: {mae:.4f}")
    return mse,mae,pred, y

if __name__ == '__main__':
    list_mse_s = []
    list_mae_s = []
    list_mse_l = []
    list_mae_l = []
    for i in range(5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # 参数
        input_len = 90
        pred_len_short = 90
        pred_len_long = 365
        batch_size = 64
        hidden_dim = 64
        stride = 1

        # 读入数据
        logger.info("Loading and preprocessing data...")
        train_df = load_and_preprocess("./data/train.csv")
        test_df = load_and_preprocess("./data/test.csv")

        train_scaled, test_scaled, scaler = normalize_data(train_df, test_df)

        # 短期预测数据准备
        logger.info("Preparing short-term forecast data...")
        X_train_s, y_train_s = create_sequences(train_scaled, input_len, pred_len_short, stride)
        X_test_s, y_test_s = create_sequences(test_scaled, input_len, pred_len_short, stride)

        train_loader_s = DataLoader(TensorDataset(
            torch.tensor(X_train_s, dtype=torch.float32),
            torch.tensor(y_train_s, dtype=torch.float32)
        ), batch_size=batch_size, shuffle=True)

        model_s = LSTMForecast(input_dim=train_scaled.shape[1], hidden_dim=hidden_dim, output_len=pred_len_short)
        logger.info("Training short-term forecast model...")
        train_model(model_s, train_loader_s)

        mse_s, mae_s, pred_s, true_s = evaluate(model_s, X_test_s, y_test_s, scaler)
        plot_prediction(pred_s, true_s, "Short-Term Power Forecast (90 days)", "LSTM_short_term_forecast")
        # pred_s:516

        # 长期预测
        logger.info("Preparing long-term forecast data...")
        X_train_l, y_train_l = create_sequences(train_scaled, input_len, pred_len_long, stride)
        X_test_l, y_test_l = create_sequences(test_scaled, input_len, pred_len_long, stride)

        train_loader_l = DataLoader(TensorDataset(
            torch.tensor(X_train_l, dtype=torch.float32),
            torch.tensor(y_train_l, dtype=torch.float32)
        ), batch_size=batch_size, shuffle=True)

        model_l = LSTMForecast(input_dim=train_scaled.shape[1], hidden_dim=hidden_dim, output_len=pred_len_long)
        logger.info("Training long-term forecast model...")
        train_model(model_l, train_loader_l)

        mse_l, mae_l, pred_l, true_l = evaluate(model_l, X_test_l, y_test_l, scaler)
        plot_prediction(pred_l, true_l, "Long-Term Power Forecast (365 days)", "LSTM_long_term_forecast")
        # pred_l:241
        logger.info("All tasks completed!")
        list_mse_s.append(mse_s)
        list_mae_s.append(mae_s)
        list_mse_l.append(mse_l)
        list_mae_l.append(mae_l)

    import numpy as np

    mse_s_mean = np.mean(list_mse_s)
    mse_s_std_dev = np.std(list_mse_s, ddof=1)
    mae_s_mean = np.mean(list_mae_s)
    mae_s_std_dev = np.std(list_mae_s, ddof=1)
    mse_l_mean = np.mean(list_mse_l)
    mse_l_std_dev = np.std(list_mse_l, ddof=1)
    mae_l_mean = np.mean(list_mae_l)
    mae_l_std_dev = np.std(list_mae_l, ddof=1)

    logger.info(f"平均值: {mse_s_mean:.4f}" + f"标准差: {mse_s_std_dev:.4f}")
    logger.info(f"平均值: {mae_s_mean:.4f}" + f"标准差: {mae_s_std_dev:.4f}")
    logger.info(f"平均值: {mse_l_mean:.4f}" + f"标准差: {mse_l_std_dev:.4f}")
    logger.info(f"平均值: {mae_l_mean:.4f}" + f"标准差: {mae_l_std_dev:.4f}")