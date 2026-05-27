"""
基于消融结果中表现最优的组合：
- Fourier 波长编码
- Residual + LayerNorm
- Dropout = 0（等效 no_dropout）
- Warmup + Cosine 调度
- 分桶高损耗加权

输出：训练曲线、散点图、预测 npz，模型权重 physics_ann_improved_no_dropout.pth
"""

import math
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(42)
np.random.seed(42)

# ---------------- 数据与预处理 ----------------
class FiberDataset(Dataset):
    def __init__(self, X, y, re_neff, re_spp):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.re_neff = torch.FloatTensor(re_neff)
        self.re_spp = torch.FloatTensor(re_spp)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.re_neff[idx], self.re_spp[idx]


def load_and_preprocess_data(filepath="dataset.xlsx", use_boxcox_transform=True):
    df = pd.read_excel(filepath)
    X = df.iloc[:, 0:9].values
    y = df.iloc[:, 12].values.reshape(-1, 1)
    re_neff = df.iloc[:, 9].values.reshape(-1, 1)
    re_spp = df.iloc[:, 11].values.reshape(-1, 1)

    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1) |
             np.isnan(re_neff).any(axis=1) | np.isnan(re_spp).any(axis=1))
    X, y, re_neff, re_spp = X[mask], y[mask], re_neff[mask], re_spp[mask]

    boxcox_transformer = None
    y_shift = 0.0
    if use_boxcox_transform:
        y_min = y.min()
        if y_min <= 0:
            y_shift = abs(y_min) + 1e-6
            y = y + y_shift
        boxcox_transformer = PowerTransformer(method="box-cox", standardize=False)
        y = boxcox_transformer.fit_transform(y)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    scaler_re_neff = StandardScaler()
    scaler_re_spp = StandardScaler()
    Xs = scaler_X.fit_transform(X)
    ys = scaler_y.fit_transform(y)
    re1s = scaler_re_neff.fit_transform(re_neff)
    re2s = scaler_re_spp.fit_transform(re_spp)

    Xtr, Xte, ytr, yte, re1tr, re1te, re2tr, re2te = train_test_split(
        Xs, ys, re1s, re2s, test_size=0.1, random_state=42
    )
    transform_info = {
        "use_boxcox_transform": use_boxcox_transform,
        "boxcox_transformer": boxcox_transformer,
        "y_shift": y_shift,
    }
    return (
        Xtr, Xte, ytr, yte,
        re1tr, re1te, re2tr, re2te,
        scaler_X, scaler_y, scaler_re_neff, scaler_re_spp,
        transform_info
    )


LOSS_COEFF = -8.686 * 2 * math.pi


def imag_to_loss_tensor(imag_tensor, wl_tensor):
    return LOSS_COEFF * (imag_tensor / (wl_tensor + 1e-8))


def imag_to_loss_numpy(imag_values, wavelengths):
    return LOSS_COEFF * (imag_values / (wavelengths + 1e-8))


def recover_imag_tensor(y_scaled, ctx):
    y_boxcox = y_scaled * ctx["y_scale"] + ctx["y_mean"]
    lam = ctx["boxcox_lambda"]
    if lam is not None:
        if abs(lam) < 1e-6:
            y_pos = torch.exp(y_boxcox)
        else:
            y_pos = torch.pow(lam * y_boxcox + 1.0, 1.0 / lam)
    else:
        y_pos = y_boxcox
    return y_pos - ctx["y_shift"]


def recover_imag_numpy(vals_scaled, scaler_y, transform_info):
    vals_boxcox = scaler_y.inverse_transform(vals_scaled)
    if transform_info and transform_info.get("use_boxcox_transform", False):
        transformer = transform_info.get("boxcox_transformer")
        if transformer is not None:
            vals_boxcox = transformer.inverse_transform(vals_boxcox)
            y_shift = transform_info.get("y_shift", 0.0)
            return vals_boxcox - y_shift
    y_shift = transform_info.get("y_shift", 0.0) if transform_info else 0.0
    return vals_boxcox - y_shift


def build_scaler_context(scaler_X, scaler_y, transform_info):
    bc = transform_info.get("boxcox_transformer")
    return {
        "y_mean": float(scaler_y.mean_[0]),
        "y_scale": float(scaler_y.scale_[0]),
        "boxcox_lambda": float(bc.lambdas_[0]) if bc is not None else None,
        "y_shift": float(transform_info.get("y_shift", 0.0)),
        "wl_mean": float(scaler_X.mean_[0]),
        "wl_scale": float(scaler_X.scale_[0]),
    }


def compute_bucket_ids(X_batch, bin_size=0.5):
    struct = X_batch[:, 1:]
    binned = torch.round(struct / bin_size).long()
    primes = struct.new_tensor([3, 5, 7, 11, 13, 17, 19, 23], dtype=torch.long)
    return (binned * primes).sum(dim=1)


def calculate_group_peak_weights(y_batch, bucket_ids,
                                 base_weight=1.0, peak_weight=3.0,
                                 peak_threshold_percentile=75, min_bucket_size=3):
    y_flat = y_batch.flatten()
    w = torch.ones_like(y_flat) * base_weight
    for b in bucket_ids.unique():
        mask = bucket_ids == b
        if mask.sum() < min_bucket_size:
            continue
        y_b = y_flat[mask]
        thr = torch.quantile(y_b, peak_threshold_percentile / 100.0)
        peak_mask = mask & (y_flat > thr)
        if peak_mask.any():
            y_peak = y_flat[peak_mask]
            y_max = y_b.max()
            w[peak_mask] = base_weight + (peak_weight - base_weight) * ((y_peak - thr) / (y_max - thr + 1e-8))
    return w.unsqueeze(1)


def weighted_mse_loss(pred, tgt, weights):
    return (weights * (pred - tgt) ** 2).mean()


# ---------------- 模型 ----------------
class FourierFeatures(nn.Module):
    def __init__(self, num_freqs=4, logspace=True):
        super().__init__()
        if logspace:
            self.register_buffer("freq_bands", 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs))
        else:
            self.register_buffer("freq_bands", torch.linspace(1.0, 2.0 ** (num_freqs - 1), num_freqs))

    def forward(self, x):
        x_exp = x[..., None] * self.freq_bands
        return torch.cat([torch.sin(x_exp), torch.cos(x_exp)], dim=-1)


class ResidualFFN(nn.Module):
    def __init__(self, dim, dropout=0.0, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.ff(self.norm(x))
        return x + out if self.use_residual else out


class ImprovedANN(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, num_layers=5,
                 fourier_freqs=4, use_fourier=True,
                 dropout=0.0, use_residual=True):
        super().__init__()
        self.use_fourier = use_fourier
        if use_fourier:
            self.fourier = FourierFeatures(num_freqs=fourier_freqs)
            wl_dim = fourier_freqs * 2
        else:
            self.fourier = None
            wl_dim = 1
        self.input_proj = nn.Linear(wl_dim + (input_dim - 1), hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualFFN(hidden_dim, dropout=dropout, use_residual=use_residual)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_dim, 3)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        wl = x[:, 0:1]
        struct = x[:, 1:]
        wl_enc = self.fourier(wl).view(wl.size(0), -1) if self.use_fourier else wl
        h = torch.cat([wl_enc, struct], dim=1)
        h = self.input_proj(h)
        for blk in self.blocks:
            h = blk(h)
        out = self.head(h)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]


# ---------------- 训练与评估 ----------------
def make_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr=1e-6, base_lr=5e-4):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return (min_lr / base_lr) + cosine * (1 - min_lr / base_lr)
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_model(model, train_loader, val_loader,
                epochs=200, lr=0.0005, lambda_im=0.1,
                use_weighted_loss=True, peak_weight=3.0, peak_threshold_percentile=75,
                scaler_context=None, use_warmup_cosine=True, warmup_frac=0.05, bin_size=0.5):
    if scaler_context is None:
        raise ValueError("scaler_context 不能为空")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    if use_warmup_cosine:
        total_steps = epochs * len(train_loader)
        warmup_steps = int(total_steps * warmup_frac)
        sched = make_warmup_cosine_scheduler(opt, warmup_steps, total_steps, min_lr=lr * 0.1, base_lr=lr)
        per_iter = True
    else:
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10, verbose=False)
        per_iter = False

    train_losses, val_losses = [], []
    for ep in range(epochs):
        model.train()
        t_total = t_main = t_aux = 0.0
        for Xb, yb, re1b, re2b in train_loader:
            Xb, yb, re1b, re2b = Xb.to(device), yb.to(device), re1b.to(device), re2b.to(device)
            opt.zero_grad()
            lp, r1p, r2p = model(Xb)
            buckets = compute_bucket_ids(Xb, bin_size=bin_size)
            if use_weighted_loss:
                w = calculate_group_peak_weights(yb, buckets, 1.0, peak_weight, peak_threshold_percentile)
                loss_main = weighted_mse_loss(lp, yb, w)
            else:
                loss_main = criterion(lp, yb)
            loss_aux = lambda_im * (criterion(r1p, re1b) + criterion(r2p, re2b))
            total = loss_main + loss_aux
            if torch.isnan(total):
                continue
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if per_iter:
                sched.step()
            t_total += total.item()
            t_main += loss_main.item()
            t_aux += loss_aux.item()
        t_total /= len(train_loader)
        t_main /= len(train_loader)
        t_aux /= len(train_loader)

        model.eval()
        v_total = 0.0
        with torch.no_grad():
            for Xb, yb, re1b, re2b in val_loader:
                Xb, yb, re1b, re2b = Xb.to(device), yb.to(device), re1b.to(device), re2b.to(device)
                lp, r1p, r2p = model(Xb)
                buckets = compute_bucket_ids(Xb, bin_size=bin_size)
                if use_weighted_loss:
                    w = calculate_group_peak_weights(yb, buckets, 1.0, peak_weight, peak_threshold_percentile)
                    loss_main = weighted_mse_loss(lp, yb, w)
                else:
                    loss_main = criterion(lp, yb)
                loss_aux = lambda_im * (criterion(r1p, re1b) + criterion(r2p, re2b))
                total = loss_main + loss_aux
                if not torch.isnan(total):
                    v_total += total.item()
        v_total /= len(val_loader)
        train_losses.append(t_total)
        val_losses.append(v_total)
        if not per_iter:
            sched.step(v_total)
        if (ep + 1) % 20 == 0:
            print(f"Epoch {ep+1}/200 | Train {t_total:.6f} (Main {t_main:.6f}, Aux {t_aux:.6f}) | Val {v_total:.6f}")

    return train_losses, val_losses


def evaluate_model(model, data_loader, scaler_X, scaler_y, scaler_re_neff, scaler_re_spp, transform_info):
    device = next(model.parameters()).device
    model.eval()
    loss_pred = []
    loss_true = []
    re1p = []
    re2p = []
    re1t = []
    re2t = []
    wls = []
    with torch.no_grad():
        for Xb, yb, re1b, re2b in data_loader:
            Xb, yb, re1b, re2b = Xb.to(device), yb.to(device), re1b.to(device), re2b.to(device)
            lp, r1p, r2p = model(Xb)
            loss_pred.extend(lp.cpu().numpy())
            loss_true.extend(yb.cpu().numpy())
            re1p.extend(r1p.cpu().numpy())
            re2p.extend(r2p.cpu().numpy())
            re1t.extend(re1b.cpu().numpy())
            re2t.extend(re2b.cpu().numpy())
            Xnp = Xb.cpu().numpy()
            wl = Xnp[:, 0:1] * scaler_X.scale_[0] + scaler_X.mean_[0]
            wls.extend(wl)

    loss_pred = np.array(loss_pred)
    loss_true = np.array(loss_true)
    re1p = np.array(re1p)
    re2p = np.array(re2p)
    re1t = np.array(re1t)
    re2t = np.array(re2t)
    wls = np.array(wls).reshape(-1, 1)

    imag_pred = recover_imag_numpy(loss_pred, scaler_y, transform_info)
    imag_true = recover_imag_numpy(loss_true, scaler_y, transform_info)
    loss_pred_cm = imag_to_loss_numpy(imag_pred, wls).flatten()
    loss_true_cm = imag_to_loss_numpy(imag_true, wls).flatten()

    re1p = scaler_re_neff.inverse_transform(re1p)
    re1t = scaler_re_neff.inverse_transform(re1t)
    re2p = scaler_re_spp.inverse_transform(re2p)
    re2t = scaler_re_spp.inverse_transform(re2t)

    metrics = {
        "loss_MSE": mean_squared_error(loss_true_cm, loss_pred_cm),
        "loss_RMSE": np.sqrt(mean_squared_error(loss_true_cm, loss_pred_cm)),
        "loss_MAE": mean_absolute_error(loss_true_cm, loss_pred_cm),
        "loss_R2": r2_score(loss_true_cm, loss_pred_cm),
        "imag_MSE": mean_squared_error(imag_true, imag_pred),
        "imag_RMSE": np.sqrt(mean_squared_error(imag_true, imag_pred)),
        "imag_MAE": mean_absolute_error(imag_true, imag_pred),
        "imag_R2": r2_score(imag_true, imag_pred),
    }
    return loss_pred_cm, loss_true_cm, re1p, re2p, re1t, re2t, metrics


def plot_results(train_losses, val_losses, loss_pred, loss_true,
                 re1p, re2p, re1t, re2t, save_dir="results_improved_no_dropout"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.scatter(loss_true, loss_pred, s=12, alpha=0.6, edgecolors="black", linewidths=0.3)
    mv, Mv = min(loss_true.min(), loss_pred.min()), max(loss_true.max(), loss_pred.max())
    plt.plot([mv, Mv], [mv, Mv], "r--")
    plt.xlabel("True Loss (1/cm)")
    plt.ylabel("Pred Loss (1/cm)")
    plt.title("Loss Prediction")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_scatter.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.scatter(re1t, re1p, s=12, alpha=0.6, edgecolors="black", linewidths=0.3)
    mv, Mv = min(re1t.min(), re1p.min()), max(re1t.max(), re1p.max())
    plt.plot([mv, Mv], [mv, Mv], "r--")
    plt.xlabel("True Re(n_eff)")
    plt.ylabel("Pred Re(n_eff)")
    plt.title("Re(n_eff) Prediction")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "re_neff_scatter.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.scatter(re2t, re2p, s=12, alpha=0.6, edgecolors="black", linewidths=0.3)
    mv, Mv = min(re2t.min(), re2p.min()), max(re2t.max(), re2p.max())
    plt.plot([mv, Mv], [mv, Mv], "r--")
    plt.xlabel("True Re(n_spp)")
    plt.ylabel("Pred Re(n_spp)")
    plt.title("Re(n_spp) Prediction")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "re_spp_scatter.png"), dpi=300)
    plt.close()


def main():
    (
        Xtr, Xte, ytr, yte, re1tr, re1te, re2tr, re2te,
        scaler_X, scaler_y, scaler_re_neff, scaler_re_spp, transform_info
    ) = load_and_preprocess_data(use_boxcox_transform=True)
    scaler_ctx = build_scaler_context(scaler_X, scaler_y, transform_info)

    train_ds = FiberDataset(Xtr, ytr, re1tr, re2tr)
    test_ds = FiberDataset(Xte, yte, re1te, re2te)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = ImprovedANN(
        input_dim=9, hidden_dim=64, num_layers=5,
        fourier_freqs=4, use_fourier=True,
        dropout=0.0, use_residual=True
    )
    print(model)
    print(f"参数总数: {sum(p.numel() for p in model.parameters()):,}")

    train_losses, val_losses = train_model(
        model, train_loader, test_loader,
        epochs=200, lr=0.0005, lambda_im=0.1,
        use_weighted_loss=True, peak_weight=3.0, peak_threshold_percentile=75,
        scaler_context=scaler_ctx, use_warmup_cosine=True, warmup_frac=0.05, bin_size=0.5
    )

    (loss_pred_cm, loss_true_cm, re1p, re2p, re1t, re2t, metrics) = evaluate_model(
        model, test_loader, scaler_X, scaler_y, scaler_re_neff, scaler_re_spp, transform_info
    )
    print("\n评估指标：")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    plot_results(train_losses, val_losses,
                 loss_pred_cm, loss_true_cm,
                 re1p, re2p, re1t, re2t,
                 save_dir="results_improved_no_dropout")

    np.savez(os.path.join("results_improved_no_dropout", "predictions.npz"),
             loss_pred=loss_pred_cm, loss_true=loss_true_cm,
             re_neff_pred=re1p, re_spp_pred=re2p,
             re_neff_true=re1t, re_spp_true=re2t)

    torch.save(model.state_dict(), "ann_improved_no_dropout.pth")
    print("训练完成，模型已保存为 ann_improved_no_dropout.pth，结果存于 results_improved_no_dropout/")


if __name__ == "__main__":
    main()

