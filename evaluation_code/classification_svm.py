import json
import gzip
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
import matplotlib.pyplot as plt
import sys

iter = sys.argv[1]
feat_filter = sys.argv[2]

def load_data(filename):
    with gzip.open(filename, "rt", encoding="utf-8") as f:
        data = json.load(f)
    split = filename.split("/")[-1].split("_")[0]
    feature_labels = data["features"]

    if feat_filter == "all":
        features_to_remove = []
    elif feat_filter == "filter":
        with open("TO_REMOVE.txt") as f:
            features_to_remove = f.read().splitlines()

    indexes_to_remove = [feature_labels.index(i) for i in features_to_remove]
    feature_labels = np.delete(feature_labels, indexes_to_remove)
    X = [x for x in data[f"X_{split}"]]  # x
    X = np.delete(X, indexes_to_remove, axis=1)
    y = data[f"y_{split}"]  # y
    return X, y, feature_labels

# def load_data(filename):
#     with gzip.open(filename, "rt", encoding="utf-8") as f:
#         data = json.load(f)
#     split = filename.split("/")[-1].split("_")[0]
#     feature_labels = data["features"]

#     # Specify the features to keep
#     features_to_keep = ["ttr_lemma_chunks_100", "ttr_lemma_chunks_200", "ttr_form_chunks_100", "ttr_form_chunks_200", "lexical_density"]
#     indexes_to_keep = [feature_labels.index(i) for i in features_to_keep if i in feature_labels]

#     # Filter the feature labels and data
#     feature_labels = np.array(feature_labels)[indexes_to_keep]
#     X = [x for x in data[f"X_{split}"]]
#     X = np.array(X)[:, indexes_to_keep]
#     y = data[f"y_{split}"]
    
#     return X, y, feature_labels


X_train, y_train, feature_labels = load_data(f"iter/{iter}/train_data.json.gz")
X_val, y_val, feature_labels = load_data(f"iter/{iter}/val_data.json.gz")
X_test, y_test, feature_labels = load_data(f"iter/{iter}/test_data.json.gz")

model = make_pipeline(MinMaxScaler(), LinearSVC(random_state=42, max_iter=1000, dual=False))

model.fit(X_train, y_train)

y_val_pred = model.predict(X_val)
print("Validation Set Performance:")
print(classification_report(y_val, y_val_pred, digits=4))

iter = iter+f"/{feat_filter}" if feat_filter != "filter" else iter

y_test_pred = model.predict(X_test)
print("Test Set Performance:")
results = classification_report(y_test, y_test_pred, digits=4, output_dict=True)
with open(f"iter/{iter}/svm_res.json", "w") as f:
    json.dump(results, f)
print(classification_report(y_test, y_test_pred, digits=4))


def plot_feature_importance(model, feature_labels):
    # Extract feature importance from the SVC model's coefficients
    coefs = model.named_steps['linearsvc'].coef_[0]

    # Create a DataFrame for easier manipulation
    importance_df = pd.DataFrame({
        "Feature": feature_labels,
        "Importance": coefs
    })

    # Split into positive and negative coefficients, sort each separately
    positive_importance = importance_df[importance_df["Importance"] > 0].sort_values(by="Importance", ascending=False)
    negative_importance = importance_df[importance_df["Importance"] < 0].sort_values(by="Importance")

    # Concatenate positive and negative for plotting
    sorted_importance_df = pd.concat([negative_importance, positive_importance])

    # Plot as a horizontal bar chart
    plt.figure(figsize=(10, 20))
    plt.barh(sorted_importance_df["Feature"], sorted_importance_df["Importance"], color='skyblue')
    plt.xlabel("Coefficient (Feature Importance)")
    plt.title("Feature coefficients from SVM")
    plt.gca().invert_yaxis()  # Highest importance at the top
    plt.tight_layout()
    plt.savefig(f"iter/{iter}/feat_importance_svc.png", dpi = 300)
    plt.show()

    sorted_importance_df.to_csv(f"iter/{iter}/svm_coefs.csv", index=False)


plot_feature_importance(model, feature_labels)