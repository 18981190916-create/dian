"""Level 1: 单张图片推理脚本
用法:
    python infer.py                      # 对测试集第 0 张图推理
    python infer.py samples/digit7.png   # 对指定图片推理
"""
import sys
from pathlib import Path

import torch
from torchvision import datasets, transforms
from PIL import Image, ImageOps

LEVEL1 = Path(__file__).resolve().parent     # 与 train.py 同款锚定
ROOT = LEVEL1.parent
DATA_DIR = ROOT / "data"
CKPT = LEVEL1 / "mlp_mnist.pt"

sys.path.insert(0, str(LEVEL1))              # 保证能找到 model.py
from model import MLP

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def predict(tensor):
    """输入 [1,1,28,28]，返回 (预测数字, 10 类概率, 10 类得分)"""
    model = MLP()                                             # ① 同结构空模型
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))  # ② 灌入权重
    model = model.to(DEVICE)                                  # ③ 上设备
    model.eval()                                              # ④ 推理模式
    with torch.no_grad():                                     # ⑤ 不算梯度
        scores = model(tensor.to(DEVICE))                     # ⑥ 前向 → [1,10] logits
    probs = torch.softmax(scores, dim=1)                      # ⑦ logits → 概率
    return scores.argmax(1).item(), probs[0], scores[0]        # ⑧ 预测类别 + 概率 + 原始得分


def load_from_testset(idx=0):
    """从 MNIST 测试集取第 idx 张（带标准答案）"""
    testset = datasets.MNIST(root=str(DATA_DIR), train=False, download=True,
                             transform=transforms.ToTensor())
    img, label = testset[idx]
    return img.unsqueeze(0), label                            # [1,28,28] → [1,1,28,28]


def load_from_file(path):
    """读取外部图片，补平与 MNIST 的三处差异"""
    img = Image.open(path).convert("L")           # ① 彩色/多通道 → 单通道灰度
    img = ImageOps.invert(img)                    # ② 白底黑笔 → 黑底白笔（对齐 MNIST）
    img = img.resize((28, 28), Image.LANCZOS)     # ③ 缩放 + 保边插值
    return transforms.ToTensor()(img).unsqueeze(0)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        x, label = load_from_file(sys.argv[1]), "未知"
    else:
        x, label = load_from_testset(0)

    pred, probs, logits = predict(x)
    print(f"真实标签: {label}")
    print(f"模型预测: {pred}")
    print(f"10 类得分: {[round(v, 2) for v in logits.tolist()]}")
    print(f"各类概率: {[f'{p:.2%}' for p in probs.tolist()]}")
    print(f"最高置信度: {probs[pred].item():.2%}")
