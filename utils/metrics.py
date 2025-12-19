import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def evaluate_and_plot(y_true, y_pred, class_names, save_dir=None, title_prefix=""):
    """
    Computes:
      - classification report
      - confusion matrix
      - class-wise accuracy
      - saves confusion matrix + bar chart (optional)

    Args:
        y_true (list or np.array): true labels
        y_pred (list or np.array): predicted labels
        class_names (list): ["Car", "Human", "Noise"]
        save_dir (str): directory to save plots (optional)
        title_prefix (str): prefix for plot titles
    """

    # -------------------------
    # Classification report
    # -------------------------
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(f"\n{title_prefix} Classification Report:\n")
    print(report)

    # -------------------------
    # Confusion Matrix
    # -------------------------
    cm = confusion_matrix(y_true, y_pred)

    # -------------------------
    # Class-wise Accuracy
    # -------------------------
    class_accuracy = {
        class_names[i]: cm[i, i] / cm[i].sum() for i in range(len(class_names))
    }

    print("\nClass-wise Accuracy:")
    for cls, acc in class_accuracy.items():
        print(f"{cls}: {acc:.4f}")

    # -------------------------
    # Plotting (optional)
    # -------------------------
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        # Confusion Matrix plot
        plt.figure(figsize=(6, 5))
        plt.imshow(cm)
        plt.title(f"{title_prefix} Confusion Matrix")
        plt.colorbar()
        plt.xticks(range(len(class_names)), class_names)
        plt.yticks(range(len(class_names)), class_names)

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )

        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
        plt.close()

        # Class Accuracy bar chart
        plt.figure(figsize=(6, 4))
        plt.bar(class_accuracy.keys(), class_accuracy.values())
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
        plt.title(f"{title_prefix} Class-wise Accuracy")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "class_accuracy.png"))
        plt.close()

        print(f"\nPlots saved in: {save_dir}")

    return class_accuracy, cm
