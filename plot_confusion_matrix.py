import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_confusion_matrix(predictions_csv_path, class_names, save_path):
    """
    绘制归一化的混淆矩阵（不显示数字）
    :param predictions_csv_path: 包含真实标签和预测标签的 CSV 文件路径
    :param class_names: 类别名称列表
    :param save_path: 混淆矩阵图片保存路径
    """
    # 读取预测结果
    predictions_df = pd.read_csv(predictions_csv_path)
    true_labels = predictions_df["true_label"]
    pred_labels = predictions_df["pred_label"]

    # 计算混淆矩阵并归一化
    cm = confusion_matrix(true_labels, pred_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)  # 使用蓝色渐变色表
    plt.title("Normalized Confusion Matrix (without numbers)", fontsize=14)
    plt.colorbar(label="Proportion")  # 添加颜色条并设置标签

    # 设置轴标签
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=10)  # X轴类别标签
    plt.yticks(tick_marks, class_names, fontsize=10)  # Y轴类别标签

    # 设置轴标签标题
    plt.xlabel("Predicted Labels", fontsize=12)
    plt.ylabel("True Labels", fontsize=12)

    # 保存图片
    plt.tight_layout()  # 自动调整布局
    plt.savefig(save_path, bbox_inches="tight")  # 保存时去掉多余空白
    plt.close()

    print(f"confusion matrix saved to {save_path}")


if __name__ == "__main__":
    # 示例用法
    predictions_csv_path = "results/csv/predictions.csv"  # 替换为你的预测结果路径
    class_names = [str(i) for i in range(43)]  # 替换为你的类别名称
    save_path = "results/pics/confusion_matrix.png"
    plot_confusion_matrix(predictions_csv_path, class_names, save_path)