import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize


def plot_roc_curve_optimized(predictions_csv_path, num_classes, save_path, top_k=5):
    """
    优化后的 ROC 曲线绘制：包括关键类别和平均 ROC 曲线
    :param predictions_csv_path: 包含真实标签和预测概率的 CSV 文件路径
    :param num_classes: 类别数量
    :param save_path: ROC 曲线图片保存路径
    :param top_k: 绘制表现最好的和最差的类别数量
    """
    # 读取预测结果
    predictions_df = pd.read_csv(predictions_csv_path)
    true_labels = predictions_df["true_label"].to_numpy()
    pred_probs = np.array([np.fromstring(row[1:-1], sep=' ') for row in predictions_df["pred_probs"]])

    # 将真实标签进行 one-hot 编码
    true_labels_one_hot = label_binarize(true_labels, classes=range(num_classes))

    # 计算每个类别的 ROC 曲线和 AUC
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels_one_hot[:, i], pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算宏平均 ROC 曲线和加权平均 ROC 曲线
    # Macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])  # 插值
    mean_tpr /= num_classes
    macro_fpr = all_fpr
    macro_tpr = mean_tpr
    macro_auc = auc(macro_fpr, macro_tpr)

    # Weighted-average
    weights = np.sum(true_labels_one_hot, axis=0) / true_labels_one_hot.shape[0]
    weighted_auc = np.sum([roc_auc[i] * weights[i] for i in range(num_classes)])

    # 找到表现最好的和最差的类别
    sorted_auc = sorted(roc_auc.items(), key=lambda x: x[1], reverse=True)
    best_classes = sorted_auc[:top_k]
    worst_classes = sorted_auc[-top_k:]

    # 绘制 ROC 曲线
    plt.figure(figsize=(12, 8))

    # 绘制宏平均 ROC 曲线
    plt.plot(macro_fpr, macro_tpr, label=f"Macro-average (AUC = {macro_auc:.2f})", color="navy", linestyle="--", linewidth=2)

    # 绘制加权平均 ROC 曲线
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.title(f"ROC Curve (Top {top_k} Classes and Averages)", fontsize=16)
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)

    # 绘制表现最好的类别
    for i, auc_score in best_classes:
        plt.plot(fpr[i], tpr[i], label=f"Best Class {i} (AUC = {auc_score:.2f})")

    # 绘制表现最差的类别
    for i, auc_score in worst_classes:
        plt.plot(fpr[i], tpr[i], label=f"Worst Class {i} (AUC = {auc_score:.2f})")

    # 图例和保存
    plt.legend(loc="lower right", fontsize=12)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"ROC curve saved to {save_path}")


if __name__ == "__main__":
    # 示例用法
    predictions_csv_path = "results/csv/predictions.csv"  # 替换为你的预测结果路径
    num_classes = 43  # 替换为你的类别数量
    save_path = "results/pics/optimized_ROC_curve.png"
    plot_roc_curve_optimized(predictions_csv_path, num_classes, save_path, top_k=3)