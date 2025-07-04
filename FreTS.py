import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from utils import load_and_preprocess, normalize_data, create_sequences, setup_logging, plot_prediction
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = setup_logging(modle='FreTS')

class FreTSModel(nn.Module):
    def __init__(self, configs):
        super(FreTSModel, self).__init__()
        self.embed_size = 128 #embed_size
        self.hidden_size = 256 #hidden_size
        self.pre_length = configs.pred_len
        self.feature_size = configs.enc_in #channels
        self.seq_length = configs.seq_len
        self.channel_independence = configs.channel_independence
        self.sparsity_threshold = 0.01
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.seq_length * self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )

    # dimension extension
    def tokenEmb(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        # N*T*1 x 1*D = N*T*D
        y = self.embeddings
        return x * y

    # frequency temporal learner
    def MLP_temporal(self, x, B, N, L):
        # [B, N, T, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho') # FFT on L dimension
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=self.seq_length, dim=2, norm="ortho")
        return x

    # frequency channel learner
    def MLP_channel(self, x, B, N, L):
        # [B, N, T, D]
        x = x.permute(0, 2, 1, 3)
        # [B, T, N, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho') # FFT on N dimension
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")
        x = x.permute(0, 2, 1, 3)
        # [B, N, T, D]
        return x

    # frequency-domain MLPs
    # dimension: FFT along the dimension, r: the real part of weights, i: the imaginary part of weights
    # rb: the real part of bias, ib: the imaginary part of bias
    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        o1_real = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)
        o1_imag = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)

        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, r) - \
            torch.einsum('bijd,dd->bijd', x.imag, i) + \
            rb
        )

        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, r) + \
            torch.einsum('bijd,dd->bijd', x.real, i) + \
            ib
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        B, T, N = x.shape
        # embedding x: [B, N, T, D]
        x = self.tokenEmb(x)
        bias = x
        # [B, N, T, D]
        if self.channel_independence == '1':
            x = self.MLP_channel(x, B, N, T)
        # [B, N, T, D]
        x = self.MLP_temporal(x, B, N, T)
        x = x + bias
        x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)
        return x

def get_config(input_len, pred_len, enc_in, channel_independence='1'):
    class Config:
        def __init__(self):
            self.seq_len = input_len
            self.pred_len = pred_len
            self.enc_in = enc_in
            self.channel_independence = channel_independence
    return Config()

def evaluate(model, X, y, scaler):
    model.eval()
    with torch.no_grad():
        X = torch.tensor(X, dtype=torch.float32).to(device)
        pred = model(X).cpu().numpy()[:, :, 0]

    # mse = nn.MSELoss()(torch.tensor(pred), torch.tensor(y)).item()
    # mae = nn.L1Loss()(torch.tensor(pred), torch.tensor(y)).item()
    mse = mean_squared_error(y, pred)
    mae = mean_absolute_error(y, pred)

    gmin, gmax = scaler.data_min_[0], scaler.data_max_[0]
    pred = pred * (gmax - gmin) + gmin
    y = y* (gmax - gmin) + gmin

    logger.info(f"MSE: {mse:.4f}, MAE: {mae:.4f}")
    return mse,mae,pred, y

# --- 主流程 ---
if __name__ == "__main__":
    list_mse_s = []
    list_mae_s = []
    list_mse_l = []
    list_mae_l = []
    for i in range(5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 参数
        input_len = 90
        output_lens = [90, 365]  # 短期与长期
        batch_size, epochs = 64, 10

        # 数据加载
        train_df = load_and_preprocess("./data/train.csv")
        test_df = load_and_preprocess("./data/test.csv")

        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_df)
        test_scaled = scaler.transform(test_df)

        for pred_len in output_lens:
            X_tr, y_tr = create_sequences(train_scaled, input_len, pred_len)
            X_te, y_te = create_sequences(test_scaled, input_len, pred_len)

            train_loader = DataLoader(TensorDataset(
                torch.tensor(X_tr, dtype=torch.float32),
                torch.tensor(y_tr, dtype=torch.float32)
            ), batch_size=batch_size, shuffle=True)

            config = get_config(input_len, pred_len, train_df.shape[1])
            model = FreTSModel(config).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()

            logger.info(f"\nTraining FreTS model for pred_len={pred_len}")
            for ep in range(epochs):
                model.train()
                total = 0
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)[:, :, 0]  # 只取第一个通道，对应 global_active_power
                    loss = criterion(pred, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total += loss.item()
                logger.info(f"Epoch {ep + 1}/{epochs} - Loss: {total / len(train_loader):.4f}")

            mse, mae, pred, true = evaluate(model, X_te, y_te, scaler)
            plot_prediction(pred, true, f"FreTS Prediction ({pred_len} days)", f"FreTS_prediction_{pred_len}_days")
            if pred_len==90:
                list_mse_s.append(mse)
                list_mae_s.append(mae)
            else:
                list_mse_l.append(mse)
                list_mae_l.append(mae)
    import numpy as np

    mse_s_mean = np.mean(list_mse_s)
    mse_s_std_dev = np.std(list_mse_s, ddof=1)
    mae_s_mean = np.mean(list_mae_s)
    mae_s_std_dev = np.std(list_mae_s, ddof=1)
    mse_l_mean = np.mean(list_mse_l)
    mse_l_std_dev = np.std(list_mse_l, ddof=1)
    mae_l_mean = np.mean(list_mae_l)
    mae_l_std_dev = np.std(list_mae_l, ddof=1)

    logger.info(f"平均值: {mse_s_mean:.4f}"+f"标准差: {mse_s_std_dev:.4f}")
    logger.info(f"平均值: {mae_s_mean:.4f}" + f"标准差: {mae_s_std_dev:.4f}")
    logger.info(f"平均值: {mse_l_mean:.4f}" + f"标准差: {mse_l_std_dev:.4f}")
    logger.info(f"平均值: {mae_l_mean:.4f}" + f"标准差: {mae_l_std_dev:.4f}")