# ============================================================
# Level 1 - 第 1 步：认识 MNIST 数据
# 目标：下载 MNIST 数据集，看看"手写数字"在代码里长什么样
# 跑完你应该能回答：一张图片在 Python 里是什么？
# ============================================================

# torchvision 是 PyTorch 处理图像的配套库
# datasets 里有 MNIST 数据集的下载和加载功能
from torchvision import datasets

# transforms 负责预处理：这里把图片转成 Tensor（PyTorch 的数据容器）
from torchvision import transforms

import matplotlib
# 不弹窗口，直接把图存成文件（在服务器上跑代码的标准做法）
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- 1. 下载训练集 ----------
# root="./data"      : 存到当前目录的 data 文件夹
# train=True         : 要训练集（MNIST 分训练集 6 万张 + 测试集 1 万张）
# download=True      : 本地没有就自动下载（约 10MB，只需下载一次）
# transforms.ToTensor(): 把图片转成 Tensor，像素值从 0~255 缩到 0~1
train_set = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)

# ---------- 2. 看看这个数据集的基本信息 ----------
print("训练集一共几张图:", len(train_set))
img, label = train_set[0]        # 取出第 0 张图
print("单张图片的形状:", img.shape)   # [1, 28, 28] = 1通道灰度图, 28x28 像素
print("这张图的真实答案:", label)     # 一个数字 0~9

# ---------- 3. 把前 10 张图存成一张图看看 ----------
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    image, answer = train_set[i]
    ax.imshow(image.squeeze(), cmap="gray")   # squeeze: 去掉通道维度才能显示
    ax.set_title(f"答案: {answer}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("preview_mnist.png", dpi=120)
print("已保存 preview_mnist.png，打开看看这 10 张手写数字")
