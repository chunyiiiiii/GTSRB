import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import os
from datetime import datetime
from torchvision.models import ResNet18_Weights, ResNet34_Weights
from data_loader import create_loaders  # 导入之前的数据加载代码


# 训练配置
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 43
    num_epochs = 30
    lr = 0.0005  # 全连接层学习率
    conv_lr = 0.0001  # 卷积层学习率（后续解冻时使用）
    weight_decay = 1e-4  # 权重衰减系数
    label_smoothing = 0.1  # 新增标签平滑

    batch_size = 32
    save_dir = "results"
    model_name = "resnet18_gtsrb"
    early_stop_patience = 5
    unfreeze_epoch = 5  # 第5个epoch开始解冻
    unfreeze_layers = False
    pct_start = 0.3  # 学习率上升期占比（前30%的step学习率上升）

def initialize_model(config):
    """初始化预训练模型"""
    model = models.resnet34(weights = ResNet34_Weights.IMAGENET1K_V1)

    # 第一阶段：只训练全连接层
    for param in model.parameters():
        param.requires_grad = False

        # 增强分类头
        model.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, config.num_classes)
        )

    return model.to(config.device)


def train_epoch(model, loader, criterion, optimizer, config, scheduler):
    """训练单个epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels in loader:
        inputs = inputs.to(config.device)
        labels = labels.to(config.device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # 添加梯度裁剪（在backward之后，step之前）
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,  # 梯度最大范数阈值
            norm_type=2  # L2范数
        )

        optimizer.step()
        scheduler.step()  # 每个batch更新学习率

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")

    return epoch_loss, epoch_acc, epoch_f1


def validate_epoch(model, loader, criterion, config, mode="validate"):
    """验证或测试单个epoch"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            if mode == "test":
                # 测试模式下返回 (inputs, labels, paths)
                inputs, labels, _ = batch  # 忽略文件名
            else:
                # 验证模式下只返回 (inputs, labels)
                inputs, labels = batch

            inputs = inputs.to(config.device)
            labels = labels.to(config.device)

            outputs = model(inputs)

            if mode != "test":  # 仅在验证模式下计算损失
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if mode == "test":
        # 测试模式返回评估指标
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average="macro")
        return epoch_acc, epoch_f1, all_preds
    else:
        # 验证模式返回损失和评估指标
        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_precision = precision_score(all_labels, all_preds, average='macro')
        epoch_recall = recall_score(all_labels, all_preds, average='macro')
        epoch_f1 = f1_score(all_labels, all_preds, average="macro")
        return epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1


def save_best_checkpoint(model, optimizer, epoch, metrics, config,is_best=False):
    """只保存最佳模型和训练状态"""
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_metrics": metrics
    }
    bestname = f"{config.model_name}_best.pth"
    torch.save(state, os.path.join(config.save_dir, 'model', bestname))


def main():
    # 初始化配置
    config = Config()
    os.makedirs(config.save_dir, exist_ok=True)

    DATA_DIR = "input/GTSRB"  # 修改为实际路径
    loaders = create_loaders(DATA_DIR, batch_size=32)

    # 准备数据
    train_loader = loaders['train']
    val_loader = loaders['val']

    # 初始化模型
    model = initialize_model(config)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config.label_smoothing  # 缓解过拟合
    )
    optimizer = optim.AdamW([
        {'params': model.fc.parameters(), 'lr': config.lr},
    ], weight_decay=config.weight_decay)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[config.lr],
        steps_per_epoch=len(train_loader),
        epochs=config.num_epochs,
        pct_start=config.pct_start,
        anneal_strategy='cos'
    )

    # 初始化记录文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_df = pd.DataFrame(columns=[
        "epoch", "train_loss", "train_acc", "train_f1",
        "val_loss", "val_acc", "val_precision", "val_recall", "val_f1"
    ])

    best_f1 = 0.0
    early_stop_counter = 0

    # 训练循环
    for epoch in range(config.num_epochs):
        # 检查是否需要解冻
        if epoch == config.unfreeze_epoch and not config.unfreeze_layers:
            print(f"Unfreezing layers at epoch {epoch + 1}")
            for param in model.layer3.parameters():
                param.requires_grad = True
            for param in model.layer4.parameters():
                param.requires_grad = True
            config.unfreeze_layers = True

            optimizer = optim.AdamW([
                {'params': model.fc.parameters(), 'lr': config.lr},
                {'params': model.layer3.parameters(), 'lr': config.conv_lr},
                {'params': model.layer4.parameters(), 'lr': config.conv_lr}
            ], weight_decay=config.weight_decay)

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[config.lr, config.conv_lr, config.conv_lr],
                steps_per_epoch=len(train_loader),
                epochs=config.num_epochs - epoch,
                pct_start=config.pct_start,
                anneal_strategy='cos'
            )

        # 打印解冻状态
        print(f"Epoch {epoch + 1} Trainable Parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  {name}")

        # 训练阶段
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, config, scheduler)

        # 验证阶段
        val_loss, val_acc, val_precision, val_recall, val_f1 = validate_epoch(
            model, val_loader, criterion, config)

        # 记录指标
        new_row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1
        }
        log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_best_checkpoint(model, optimizer, epoch + 1, new_row, config)
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # 早停检查
        if early_stop_counter >= config.early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        # 打印进度
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")
        print("-" * 60)

    # 保存日志文件
    log_df.to_csv(os.path.join(config.save_dir, 'csv', f"training_log.csv"), index=False)

if __name__ == "__main__":
    main()