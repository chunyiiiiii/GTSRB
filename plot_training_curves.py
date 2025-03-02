import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_curves(log_csv_path, save_dir):
    """
    绘制训练损失、准确率和 F1 分数曲线
    :param log_csv_path: 训练日志 CSV 文件路径
    :param save_dir: 保存图表的目录
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 读取训练日志
    log_df = pd.read_csv(log_csv_path)

    # 绘制 Loss 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(log_df["epoch"], log_df["train_loss"], label="Train Loss")
    plt.plot(log_df["epoch"], log_df["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    # 绘制 Accuracy 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(log_df["epoch"], log_df["train_acc"], label="Train Accuracy")
    plt.plot(log_df["epoch"], log_df["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"))
    plt.close()

    # 绘制 F1 Score 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(log_df["epoch"], log_df["train_f1"], label="Train F1 Score")
    plt.plot(log_df["epoch"], log_df["val_f1"], label="Validation F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Training and Validation F1 Score")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "f1_curve.png"))
    plt.close()

    print(f"Training curves saved to {save_dir}")


if __name__ == "__main__":
    # 示例用法
    log_csv_path = "results/csv/training_log.csv"  # 替换为你的训练日志路径
    save_dir = "results/pics"
    plot_training_curves(log_csv_path, save_dir)