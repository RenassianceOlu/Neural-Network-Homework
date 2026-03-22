import copy
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

BASE_SEED = 42
random.seed(BASE_SEED)
np.random.seed(BASE_SEED)
torch.manual_seed(BASE_SEED)

# 是否启用模型集成
USE_ENSEMBLE = True
ENSEMBLE_SEEDS = [11, 22, 33, 44, 55]


# ================= 1. 数据准备 =================
file_path = r"e:\神经网络\Concrete_Data_Yeh.csv"
data = pd.read_csv(file_path)

print("数据读取成功！")
print(f"数据维度: {data.shape}")

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=BASE_SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=BASE_SEED
)

print(f"训练集: {X_train.shape[0]} | 验证集: {X_val.shape[0]} | 测试集: {X_test.shape[0]}")


# ================= 2. 相关性分析 =================
plt.figure(figsize=(10, 8))
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("特征相关性分析")
plt.savefig("01_特征相关性分析.png", dpi=300, bbox_inches="tight")
plt.show()


# ================= 3. 数据预处理 =================
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(X_train)
X_val_scaled = x_scaler.transform(X_val)
X_test_scaled = x_scaler.transform(X_test)

y_train_scaled = y_scaler.fit_transform(y_train)
y_val_scaled = y_scaler.transform(y_val)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True
)


# ================= 4. 模型定义 =================
class ConcreteStrengthNet(nn.Module):
    def __init__(self, input_dim, hidden1=192, hidden2=96, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_one_model(seed, verbose=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = ConcreteStrengthNet(input_dim=X.shape[1], hidden1=192, hidden2=96, dropout=0.05)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=25, min_lr=5e-6
    )

    max_epochs = 1200
    early_stop_patience = 120
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    train_losses = []
    val_losses = []

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor)
            val_loss = criterion(val_pred, y_val_tensor).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-7:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch + 1) % 100 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch + 1}/{max_epochs}] | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}"
            )

        if no_improve >= early_stop_patience:
            if verbose:
                print(f"早停触发：在第 {epoch + 1} 轮停止训练。")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        pred_scaled = model(X_test_tensor).cpu().numpy()
    pred = y_scaler.inverse_transform(pred_scaled).reshape(-1)

    return model, pred, train_losses, val_losses, best_val_loss


# ================= 5. 训练与预测 =================
if USE_ENSEMBLE:
    print("开始训练（集成模式）...")
    all_preds = []
    first_train_losses = None
    first_val_losses = None

    for i, seed in enumerate(ENSEMBLE_SEEDS):
        verbose = i == 0
        _, pred_i, tr_losses, va_losses, best_val = train_one_model(seed=seed, verbose=verbose)
        all_preds.append(pred_i)
        print(f"子模型 seed={seed} 最佳验证损失: {best_val:.5f}")

        if first_train_losses is None:
            first_train_losses = tr_losses
            first_val_losses = va_losses

    predictions_np = np.mean(np.vstack(all_preds), axis=0)
    train_losses = first_train_losses
    val_losses = first_val_losses
else:
    print("开始训练（单模型模式）...")
    _, predictions_np, train_losses, val_losses, _ = train_one_model(seed=BASE_SEED, verbose=True)


plt.figure(figsize=(8, 6))
plt.plot(train_losses, label="Training Loss", color="teal", linewidth=2)
plt.plot(val_losses, label="Validation Loss", color="darkorange", linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE, scaled-y)")
plt.title("训练与验证损失曲线")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("02_训练损失曲线.png", dpi=300, bbox_inches="tight")
plt.show()


# ================= 6. 模型测试和评估 =================
y_test_np = y_test.reshape(-1)

mse = mean_squared_error(y_test_np, predictions_np)
mae = mean_absolute_error(y_test_np, predictions_np)
r2 = r2_score(y_test_np, predictions_np)
residual = predictions_np - y_test_np
outlier_ratio = (np.abs(residual) > 8.0).mean() * 100

print()
print("测试集评估指标:")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R^2: {r2:.4f}")
print(f"|误差| > 8 MPa 的样本占比: {outlier_ratio:.2f}%")


# ================= 7. 结果可视化 =================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(y_test_np[:50], label="真实值 (Target)", color="royalblue", marker="o")
plt.plot(predictions_np[:50], label="预测值 (Output)", color="darkorange", linestyle="--", marker="x")
plt.title("真实值 vs 预测值 (前50个测试样本)")
plt.xlabel("样本索引")
plt.ylabel("抗压强度 (MPa)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.subplot(1, 2, 2)
plt.scatter(y_test_np, predictions_np, alpha=0.6, color="seagreen")
line_min = min(y_test_np.min(), predictions_np.min())
line_max = max(y_test_np.max(), predictions_np.max())
plt.plot([line_min, line_max], [line_min, line_max], "r--", linewidth=2)
plt.xlabel("真实值 (Target)")
plt.ylabel("预测值 (Output)")
plt.title(f"整体预测效果散点图 (R^2 = {r2:.3f})")
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("03_测试集预测结果对比.png", dpi=300, bbox_inches="tight")
plt.show()
