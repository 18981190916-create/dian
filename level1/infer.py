"""Level 1: 单张图片推理脚本
用法:
    python infer.py                    # 默认对测试集第 0 张图推理
    python infer.py path/to/img.png    # 对自己指定的图片推理（如手拍/手绘的数字）
"""
import sys
import torch
from torchvision import datasets, transforms
from PIL import Image, ImageOps

from model import MLP

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def predict(tensor):
    """输入 [1,1,28,28] 的 Tensor，返回 (预测数字, 10个得分)"""
    model = MLP()                                        # ① 先造同结构的空模型
    model.load_state_dict(torch.load("mlp_mnist.pt"))    # ② 灌入训练好的权重
    model = model.to(DEVICE)
    model.eval()                                         # ③ 推理模式
    with torch.no_grad():
        scores = model(tensor.to(DEVICE))                # ④ 前向
    return scores.argmax(1).item(), scores[0]


def load_from_testset(idx=0):
    """从 MNIST 测试集取第 idx 张（自动带归一化格式）"""
    testset = datasets.MNIST(root="./data", train=False, download=True,
                             transform=transforms.ToTensor())
    img, label = testset[idx]
    return img.unsqueeze(0), label                       # 补 batch 维 -> [1,1,28,28]


def load_from_file(path):
    """读取任意外部图片，转成和 MNIST 一样的格式"""
    img = Image.open(path).convert("L")          # 转灰度
    img = ImageOps.invert(img)                   # MNIST 是黑字白底，普通照片是白字黑底，反过来
    img = img.resize((28, 28))                   # 缩放到 28x28
    tensor = transforms.ToTensor()(img)          # -> [1,28,28]，0~1
    return tensor.unsqueeze(0)


if __name__ == "__main__":
    if len(sys.argv) > 1:                        # 命令行给了图片路径
        x = load_from_file(sys.argv[1])
        label = "(未知)"
    else:                                        # 默认用测试集第 0 张
        x, label = load_from_testset(0)

    pred, scores = predict(x)
    print(f"真实标签: {label}")
    print(f"模型预测: {pred}")
    print(f"10 类得分: {[round(s, 2) for s in scores.tolist()]}")
