import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from torch import nn
from torchvision import models, transforms

# 定义类别标签（根据你的模型分类任务定义）
CLASS_LABELS = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles over 3.5 metric tons prohibited", "No entry",
    "General caution", "Dangerous curve to the left", "Dangerous curve to the right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left", "Keep right",
    "Keep left", "Roundabout mandatory", "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons"
]  # 替换为你的类别名称

# 图像预处理函数
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像大小到 256x256
        transforms.RandomCrop(224),  # 随机裁剪到 224x224，但不增加 padding
        transforms.RandomRotation(degrees=10),  # 轻微旋转，角度范围降到 ±10°
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(
            mean=[0.3402, 0.3121, 0.3214],  # GTSRB 专用均值
            std=[0.2724, 0.2608, 0.2669]  # GTSRB 专用标准差
        )
    ])
    img = Image.open(image_path).convert('RGB')  # 打开图片并转换为RGB
    return transform(img).unsqueeze(0)  # 添加 batch 维度

# 加载模型（以 ResNet50 为例）
def load_model():
    num_classes = 43
    model_path = 'results/model/resnet18_gtsrb_best.pth'

    # 加载预训练的 ResNet 模型
    model = models.resnet34(pretrained=False)
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
    return model

# 推理函数
def predict(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)  # 获取预测类别
        return CLASS_LABELS[predicted.item()]  # 返回预测类别名称

# GUI 类
class ImageClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("图片分类器")

        # 模型
        self.model = load_model()

        # 图片路径
        self.image_path = None

        # 创建界面
        self.create_widgets()

    def create_widgets(self):
        # 图片选择
        self.image_label = tk.Label(self.root, text="选择图片：")
        self.image_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.image_button = tk.Button(self.root, text="选择文件", command=self.select_image)
        self.image_button.grid(row=0, column=1, padx=10, pady=10)

        # 图片展示
        self.result_canvas = tk.Label(self.root)
        self.result_canvas.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

        # 真实类别输入
        self.true_label_label = tk.Label(self.root, text="真实类别：")
        self.true_label_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.true_label_entry = tk.Entry(self.root)
        self.true_label_entry.grid(row=2, column=1, padx=10, pady=10)

        # 预测结果
        self.predicted_label_label = tk.Label(self.root, text="预测类别：")
        self.predicted_label_label.grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.predicted_label_value = tk.Label(self.root, text="未预测")
        self.predicted_label_value.grid(row=3, column=1, padx=10, pady=10, sticky="w")

        # 运行推理按钮
        self.run_button = tk.Button(self.root, text="运行分类", command=self.run_classification)
        self.run_button.grid(row=4, column=0, columnspan=2, pady=20)

    def select_image(self):
        """选择图片文件"""
        self.image_path = filedialog.askopenfilename(
            title="选择图片文件",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png"), ("所有文件", "*.*")]
        )
        if self.image_path:
            self.display_image(self.image_path)

    def display_image(self, image_path):
        """在界面中显示图片"""
        img = Image.open(image_path)
        img = img.resize((224, 224))  # 调整图片大小
        img_tk = ImageTk.PhotoImage(img)
        self.result_canvas.config(image=img_tk)
        self.result_canvas.image = img_tk

    def run_classification(self):
        """运行分类推理"""
        if not self.image_path:
            messagebox.showerror("错误", "请先选择图片！")
            return

        # 获取真实类别
        true_label = self.true_label_entry.get()
        if not true_label:
            messagebox.showerror("错误", "请输入真实类别！")
            return

        try:
            # 预处理图片
            image_tensor = preprocess_image(self.image_path)

            # 模型推理
            predicted_label = predict(self.model, image_tensor)

            # 显示预测结果
            self.predicted_label_value.config(text=predicted_label)

            # 提示真实类别和预测类别
            messagebox.showinfo("分类结果", f"真实类别：{true_label}\n预测类别：{predicted_label}")
        except Exception as e:
            messagebox.showerror("错误", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierGUI(root)
    root.mainloop()