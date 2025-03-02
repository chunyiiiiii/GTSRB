import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class GTSRBDataset(Dataset):
    def __init__(self, root_dir, mode = 'train', transform=None):
        """
        Args:
            root_dir (str): 数据集根目录
            mode (str): 模式 ["train", "val", "test"]
            transform (callable): 数据变换
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

        # 收集所有图像路径和标注信息
        self.image_paths = []
        self.labels = []
        self.rois = []

        self._load_data()

    def _load_data(self):
        if self.mode in ["train", "val"]:
            # 加载训练集数据（后续会划分验证集）
            train_path = os.path.join(self.root_dir, "Final_Training", "Images")
            for class_dir in sorted(os.listdir(train_path)):
                class_path = os.path.join(train_path, class_dir)
                csv_file = os.path.join(class_path, f"GT-{class_dir}.csv")

                df = pd.read_csv(csv_file, delimiter=';')
                for _, row in df.iterrows():
                    self.image_paths.append(os.path.join(class_path, row['Filename']))
                    self.labels.append(row['ClassId'])
                    self.rois.append((row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2']))

        elif self.mode == "test":
            # 加载官方测试集
            test_csv = os.path.join(self.root_dir, "Final_Test", "GT-final_test.csv")
            test_img_dir = os.path.join(self.root_dir, "Final_Test", "Images")

            df = pd.read_csv(test_csv, delimiter=';')
            for _, row in df.iterrows():
                self.image_paths.append(os.path.join(test_img_dir, row['Filename']))
                self.labels.append(row['ClassId'])
                self.rois.append((row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2']))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        # ROI裁剪
        img = img.crop(self.rois[idx])

        if self.transform:
            img = self.transform(img)

        if self.mode == "test":
            return img, label, os.path.basename(self.image_paths[idx])  # 返回图像、真实标签和文件名
        else:
            return img, label


def create_loaders(data_dir, batch_size=32, val_size=0.2):
    # 数据预处理配置
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像大小到 256x256
        transforms.RandomCrop(224),  # 随机裁剪到 224x224，但不增加 padding
        transforms.RandomRotation(degrees=10),  # 轻微旋转，角度范围降到 ±10°
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(
            mean=[0.3402, 0.3121, 0.3214],  # GTSRB 专用均值
            std=[0.2724, 0.2608, 0.2669]  # GTSRB 专用标准差
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),  # 验证/测试时同样先缩放
        transforms.CenterCrop(224),  # 中心裁剪保证尺寸一致
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.3402, 0.3121, 0.3214],  # GTSRB 专用均值
            std=[0.2724, 0.2608, 0.2669]  # GTSRB 专用标准差
        )
    ])

    # 创建基础数据集
    full_train = GTSRBDataset(data_dir, mode="train", transform=train_transform)
    test_dataset = GTSRBDataset(data_dir, mode="test", transform=eval_transform)

    # 划分训练/验证集
    train_idx, val_idx = train_test_split(
        range(len(full_train)),
        test_size=val_size,
        stratify=full_train.labels,
        random_state=42
    )

    # 创建子集
    train_dataset = Subset(full_train, train_idx)
    val_dataset = Subset(full_train, val_idx)

    # 覆盖验证集的transform
    val_dataset.dataset.transform = eval_transform

    # 创建DataLoader
    loaders = {
        "train": DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True),
        "val": DataLoader(val_dataset, batch_size, shuffle=False, num_workers=4),
        "test": DataLoader(test_dataset, batch_size, shuffle=False, num_workers=4)
    }

    return loaders


if __name__ == "__main__":
    # 示例用法
    DATA_DIR = "input/GTSRB"  # 修改为实际路径
    loaders = create_loaders(DATA_DIR, batch_size=32)

    # 显示数据集大小
    print(f"Train: {len(loaders['train'].dataset)} samples")
    print(f"Validation: {len(loaders['val'].dataset)} samples")
    print(f"Test: {len(loaders['test'].dataset)} samples")

    # # 显示测试集样本
    # test_sample, test_label = next(iter(loaders["test"]))
    # print("\nTest sample shape:", test_sample.shape)
    # print("Test label:", test_label)
    # 可视化测试
    sample, label = next(iter(loaders["train"]))
    print(sample.shape)
    plt.imshow(sample[2].permute(1, 2, 0).numpy())
    plt.title(f"Label: {label[0]}")
    plt.show()