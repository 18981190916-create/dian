"""MLP 模型定义 —— 全项目共用的模型类（Level 1~4 都从这里导入）"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """784 -> 256 -> 10 的多层感知机（MNIST 手写数字识别）"""

    def __init__(self, hidden_size=256, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()                      # [N,1,28,28] -> [N,784]
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, hidden_size),             # 784 -> 256
            nn.ReLU(),                                   # 非线性
            nn.Linear(hidden_size, num_classes),         # 256 -> 10（输出原始得分）
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x


# 自测：直接运行本文件时检查模型是否正常
if __name__ == "__main__":
    m = MLP()
    X = torch.randn(64, 1, 28, 28)
    out = m(X)
    print("输出形状:", out.shape)                        # 预期 [64, 10]
    print("参数量:", sum(p.numel() for p in m.parameters()))  # 预期 203530
