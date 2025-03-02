import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os


def plot_roc_curve(predictions_csv_path, num_classes, save_path):
    """
    绘制 ROC 曲线
    :param predictions_csv_path: 包含真实标签和预测概率的 CSV 文件路径
    :param num_classes: 类别数量
    :param save_path: ROC 曲线图片保存路径
    """
    # 读取预测结果
    predictions_df = pd.read_csv(predictions_csv_path)
    true_labels = predictions_df["true_label"]
    pred_probs = np.array([np.fromstring(row[1:-1], sep=' ') for row in predictions_df["pred_probs"]])

    # 计算每个类别的 ROC 曲线和 AUC
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels == i, pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 绘制 ROC 曲线
    plt.figure(figsize=(10, 5))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_path, "ROC_curve.png"))
    plt.close()

    print(f"ROC curve saved to {save_path}")


if __name__ == "__main__":
    # 示例用法
    predictions_csv_path = "results/csv/predictions.csv"  # 替换为你的预测结果路径
    num_classes = 43  # 替换为你的类别数量
    save_path = "results/pics/"
    plot_roc_curve(predictions_csv_path, num_classes, save_path)