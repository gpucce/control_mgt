import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report


def process_dataset(dataset, llama_column):
    mage_y_gold = []
    mage_y_pred = []
    radar_y_gold = []
    radar_y_pred = []
    detectaive_y_gold = []
    detectaive_y_pred = []
    with open(dataset) as input_file:
        for doc in json.loads(input_file.read()):
            line = doc
            
            # Human texts
            mage_y_gold.append(0)
            radar_y_gold.append(0)
            detectaive_y_gold.append(0)
            
            mage_y_pred.append(0 if line['human-mage-pred'] == "human-written" else 1)
            radar_y_pred.append(0 if line['human-radar-pred'] == "human-written" else 1)
            detectaive_y_pred.append(0 if line['human-detectaive-pred'] == "human-written" or line['human-detectaive-pred'] == "human-written, machine-polished" else 1)
            
            # Llama texts
            mage_y_gold.append(1)
            radar_y_gold.append(1)
            detectaive_y_gold.append(1)
            
            mage_y_pred.append(0 if line[f'{llama_column}-mage-pred'] == "human-written" else 1)
            radar_y_pred.append(0 if line[f'{llama_column}-radar-pred'] == "human-written" else 1)
            detectaive_y_pred.append(0 if line[f'{llama_column}-detectaive-pred'] == "human-written" or line[f'{llama_column}-detectaive-pred'] == "human-written, machine-polished" else 1)
    
    plot_roc_curve(mage_y_gold, mage_y_pred, f"Mage Roc on iter {llama_column}")
    plot_roc_curve(radar_y_gold, radar_y_pred, f"Radar Roc on iter {llama_column}")
    plot_roc_curve(detectaive_y_gold, detectaive_y_pred, f"Detectaive Roc on iter {llama_column}")


def plot_roc_curve(y_gold, y_pred, title):
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
    plt.savefig(title + ".png")


if __name__  == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Path to the jsonl file containing the documents and the predictions")

    args = parser.parse_args()
    process_dataset(args.filename, "dpo-llama-1st-iter")