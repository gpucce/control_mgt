import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report
import os

def process_dataset(folder_path, output_path):
    detectors = [folder.name for folder in os.scandir(folder_path) if folder.is_dir()]
    for detector in detectors:
        iters = [folder.name for folder in os.scandir(os.path.join(folder_path, detector)) if folder.is_dir()]
        for iter in iters:
            prediction_file = os.path.join(folder_path, detector, iter, "clf_preds.csv")
            df = pd.read_csv(prediction_file)
            plot_roc_curve(df["y_true"], df['y_pred'], f"Roc Curve of {detector}: Human vs {iter}", output_path)


def plot_roc_curve(y_gold, y_pred, title, output_path):
    fpr, tpr, _ = roc_curve(y_gold, y_pred)
    roc_auc = auc(fpr, tpr)
    print(classification_report(y_true=y_gold, y_pred=y_pred))
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, title + ".png"))


if __name__  == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", "-f", help="Path to the evaluations folder", default="evaluation_code/evaluations/adversarial-dpo-iter1-filtered/2025-01-28-18-49")
    parser.add_argument("--output", "-o", default="roc/plots", help="Path where to store the plots")

    args = parser.parse_args()
    process_dataset(args.filename, args.output)