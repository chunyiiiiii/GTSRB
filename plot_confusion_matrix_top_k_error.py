import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def plot_top_k_errors(predictions_csv_path, class_names, save_path, top_k=10):
    """
    绘制错误率前 K 的混淆矩阵，并标注数值
    :param predictions_csv_path: 包含真实标签和预测标签的 CSV 文件路径
    :param class_names: 类别名称列表
    :param save_path: 混淆矩阵图片保存路径
    :param top_k: 显示错误率最高的前 K 个类别对
    """
    # 读取预测结果
    predictions_df = pd.read_csv(predictions_csv_path)
    true_labels = predictions_df["true_label"]
    pred_labels = predictions_df["pred_label"]

    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)

    # 归一化混淆矩阵（避免除以 0 的情况）
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)  # 加上很小的值防止除以 0

    # 找出误分类最多的类别
    errors = np.sum(cm, axis=1) - np.diag(cm)  # 每个类别的误分类数
    top_k_indices = np.argsort(errors)[-top_k:]  # 获取误分类最多的类别索引

    # 提取子矩阵
    cm_top_k = cm_normalized[np.ix_(top_k_indices, top_k_indices)]
    classes_top_k = [class_names[i] for i in top_k_indices]

    # 绘制混淆矩阵
    plt.figure(figsize=(12, 10))
    plt.imshow(cm_top_k, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Top {top_k} Confusion Matrix", fontsize=16)
    plt.colorbar(label="Normalized Count")  # 添加颜色条并设置标签
    tick_marks = np.arange(len(classes_top_k))
    plt.xticks(tick_marks, classes_top_k, rotation=45, fontsize=12)
    plt.yticks(tick_marks, classes_top_k, fontsize=12)

    # 在混淆矩阵上标注数值
    for i in range(len(classes_top_k)):
        for j in range(len(classes_top_k)):
            value = cm_top_k[i, j]
            plt.text(j, i, f"{value:.2f}", horizontalalignment="center",
                     color="white" if value > 0.5 else "black", fontsize=10)

    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    # 保存图片
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"Top-{top_k} error confusion matrix saved to {save_path}")


if __name__ == "__main__":
    # 示例用法
    predictions_csv_path = "results/csv/predictions.csv"  # 替换为你的预测结果路径
    class_names = [str(i) for i in range(43)]  # 替换为你的类别名称
    save_path = "results/pics/top_k_errors_confusion_matrix.png"
    plot_top_k_errors(predictions_csv_path, class_names, save_path, top_k=10)