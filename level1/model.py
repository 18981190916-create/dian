"""MLP 模型定义 —— 全项目共用的模型类(Level 1~4 都从这里导入）"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten=nn.Flatten()  # 将输入展平为一维向量
        self.linear_relu_stack=nn.Sequential(
            nn.Linear(28*28,256),  # 输入层到隐藏层1
            nn.ReLU(),      
            nn.Linear(256,10)
        )
    def forward(self,x):
        x=self.flatten(x)
        x=self.linear_relu_stack(x)
        return x


# 自测：直接运行本文件时检查模型是否正常
if __name__ == "__main__":
    m = MLP()
    X = torch.randn(64, 1, 28, 28)
    out = m(X)
    print("输出形状:", out.shape)                        # 预期 [64, 10]
    print("参数量:", sum(p.numel() for p in m.parameters()))  # 预期 203530
