"""Level 1: MLP 训练 MNIST —— 交付脚本
每次运行都从零开始训练，输出准确率、Loss 曲线和权重文件。
用法: python train.py
"""
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model import MLP

LEVEL1 = Path(__file__).resolve().parent        # 脚本所在目录 = level1/
ROOT = LEVEL1.parent                            # 项目根 = dian/
DATA_DIR = ROOT / "data"
CKPT = LEVEL1 / "mlp_mnist.pt"
CURVE = LEVEL1 / "loss_curve.png"

# ---------- 超参数 ----------
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 5
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---------- 数据 ----------
trainset = datasets.MNIST(root=str(DATA_DIR), train=True, download=True, transform=transforms.ToTensor())
testset = datasets.MNIST(root=str(DATA_DIR), train=False, download=True, transform=transforms.ToTensor())
# train_subset, val_subset = random_split(trainset, [55000, 5000])  # 防泄露：调参只看验证集
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(testset, batch_size=256, shuffle=False)

# ---------- 模型 / 损失 / 优化器（三者绑定同一模型，同建同用）----------
model = MLP().to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---------- 训练 ----------
model.train()
train_losses = []
for epoch in range(EPOCHS):
    total_loss = 0.0
    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()          # ① 清旧梯度
        scores = model(X)              # ② 前向
        loss = loss_fn(scores, y)      # ③ 打分
        loss.backward()                # ④ 反向传播
        optimizer.step()               # ⑤ 更新权重
        total_loss += loss.item() * len(y)
    avg = total_loss / len(trainset)
    train_losses.append(avg)
    print(f"epoch {epoch+1}/{EPOCHS}  平均loss: {avg:.4f}")

# ---------- 测试（只碰一次测试集）----------
model.eval()
correct = 0
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        scores = model(X)
        correct += (scores.argmax(1) == y).sum().item()
acc = correct / len(testset)
print(f"测试集准确率: {acc*100:.1f}%")

# ---------- 保存权重 + Loss 曲线 ----------
torch.save(model.state_dict(), CKPT)
print(f"权重已保存: {CKPT}")

plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, marker="o")
plt.xlabel("epoch")
plt.ylabel("training loss")
plt.title("MLP on MNIST - Loss Curve")
plt.grid(True)
plt.savefig(CURVE, dpi=150)
print(f"曲线已保存: {CURVE}")
