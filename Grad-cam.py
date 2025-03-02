import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms
from PIL import Image
import cv2
import torchvision.models as models


class GradCAM:
    def __init__(self, model, target_layer):
        """
        初始化 Grad-CAM
        :param model: PyTorch 模型
        :param target_layer: 目标卷积层，用于提取梯度和激活图
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册前向和反向钩子
        self._register_hooks()

    def _register_hooks(self):
        """
        注册前向和反向钩子函数
        """
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # 为目标层注册钩子
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        """
        生成 Grad-CAM 热力图
        :param input_tensor: 输入图像张量 (1, C, H, W)
        :param target_class: 目标类别索引。如果为 None，则使用模型预测的类别
        :return: 热力图 (H, W)
        """
        # 前向传播
        output = self.model(input_tensor)

        # 如果没有指定目标类别，则使用模型预测的类别
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        # 反向传播，计算目标类别的梯度
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()

        # 获取梯度和激活图
        gradients = self.gradients.detach().cpu().numpy()[0]  # (C, H, W)
        activations = self.activations.detach().cpu().numpy()[0]  # (C, H, W)

        # 计算权重：对梯度进行全局平均池化
        weights = np.mean(gradients, axis=(1, 2))  # (C,)

        # 计算 Grad-CAM：加权求和激活图
        cam = np.zeros(activations.shape[1:], dtype=np.float32)  # (H, W)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU 激活：去掉负值
        cam = np.maximum(cam, 0)

        # 归一化到 [0, 1]
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        return cam


def preprocess_image(image_path):
    """
    预处理输入图像
    :param image_path: 图像路径
    :return: 预处理后的张量, 原始 PIL 图像
    """
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像大小到 256x256
        transforms.RandomCrop(224),  # 随机裁剪到 224x224，但不增加 padding
        transforms.RandomRotation(degrees=10),  # 轻微旋转，角度范围降到 ±10°
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(
            mean=[0.3402, 0.3121, 0.3214],  # GTSRB 专用均值
            std=[0.2724, 0.2608, 0.2669]  # GTSRB 专用标准差
        )
    ])
    input_tensor = preprocess(image).unsqueeze(0)  # 增加 batch 维度
    return input_tensor, image


def overlay_heatmap(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    将热力图叠加到原始图像上
    :param image: 原始 PIL 图像
    :param heatmap: Grad-CAM 生成的热力图 (H, W)
    :param alpha: 热力图透明度
    :param colormap: 颜色映射表
    :return: 叠加后的图像
    """
    # 将 PIL 图像转换为 OpenCV 格式
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 将热力图缩放到与原图相同的大小
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # 应用颜色映射
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, colormap)

    # 叠加热力图
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    # 转换回 RGB 格式
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return overlay


if __name__ == "__main__":
    num_classes = 43
    model_path = 'results/model/resnet18_gtsrb_best.pth'

    # 加载预训练的 ResNet 模型
    model = models.resnet34(pretrained = False)
    model.fc = nn.Sequential(
        nn.Linear(512, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.LayerNorm(512),
        nn.GELU(),
        nn.Linear(512, num_classes)
    )
    # 加载保存的检查点
    checkpoint = torch.load(model_path)

    # 提取模型的 state_dict
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # 指定目标卷积层
    target_layer = model.layer4[2].conv1  # ResNet 最后一层卷积层

    # 初始化 Grad-CAM
    grad_cam = GradCAM(model, target_layer)

    # 加载和预处理输入图像
    image_path = "input/GTSRB/Final_Training/Images/00022/00000_00001.ppm"  # 替换为你的图像路径
    input_tensor, original_image = preprocess_image(image_path)

    # 生成 Grad-CAM 热力图
    target_class = None  # 如果你知道目标类别，可以指定类别索引；否则为 None
    cam = grad_cam.generate(input_tensor, target_class=target_class)

    # 将热力图叠加到原始图像上
    overlay = overlay_heatmap(original_image, cam)

    # 保存和展示结果
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("results/pics/grad_cam_result.png")
    plt.show()