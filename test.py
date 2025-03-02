import torch
from torch import nn
from data_loader import create_loaders
import pandas as pd
from train import validate_epoch, initialize_model

class Config:
    """测试配置"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 43
    save_dir = "results"
    model_name = "resnet18_gtsrb"
    batch_size = 64  # 测试时可以增大批次大小
    unfreeze_layers = False  # 推理时无需解冻层
    unfreeze_epoch = 5  # 任意值均可，测试时不使用

def load_best_model(config):
    model = initialize_model(config)
    checkpoint = torch.load(f"{config.save_dir}/model/{config.model_name}_best.pth")
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(config.device)


def save_predictions(model, test_loader, config):
    """保存预测结果到CSV文件"""
    model.eval()
    all_preds = []
    all_true_labels = []  # 真实标签列表
    all_files = []  # 文件名列表（假设测试数据加载器返回文件名）
    all_probs = []  # 保存预测概率

    with torch.no_grad():
        for inputs, labels, paths in test_loader:  # 测试加载器返回 (inputs, labels, paths)
            inputs = inputs.to(config.device)
            outputs = model(inputs)
            probs = nn.Softmax(dim=1)(outputs)  # 计算概率
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())  # 保存概率
            all_true_labels.extend(labels.cpu().numpy())  # 收集真实标签
            all_files.extend(paths)  # 收集文件名

    # 保存到CSV文件
    predictions_df = pd.DataFrame({
        "filename": all_files,
        "true_label": all_true_labels,
        "pred_label": all_preds,
        "pred_probs": [str(prob) for prob in all_probs]  # 转换为字符串保存
    })
    predictions_df.to_csv(f"{config.save_dir}/csv/predictions.csv", index=False)


def main():
    config = Config()
    loaders = create_loaders(data_dir="./input/GTSRB", batch_size=config.batch_size)

    # 加载模型
    model = load_best_model(config)

    # 测试模型
    test_acc, test_f1, all_preds = validate_epoch(model, loaders["test"], config, mode='test', config=config)

    # 打印测试结果
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")

    # 保存预测结果
    save_predictions(model, loaders["test"], config)


if __name__ == "__main__":
    main()