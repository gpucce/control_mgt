import json
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import export_text
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

# Load datasets
X_train, y_train, feature_labels = load_data(f"iter/{iter}/train_data.json.gz")
X_val, y_val, feature_labels = load_data(f"iter/{iter}/val_data.json.gz")
X_test, y_test, feature_labels = load_data(f"iter/{iter}/test_data.json.gz")

# Replace LinearSVC with DecisionTreeClassifier
model = make_pipeline(DecisionTreeClassifier(random_state=42, max_depth=5))

# Train the model
model.fit(X_train, y_train)

# Validate the model
y_val_pred = model.predict(X_val)
print("Validation Set Performance:")
print(classification_report(y_val, y_val_pred, digits=4))

# Test the model
y_test_pred = model.predict(X_test)
print("Test Set Performance:")
iter = iter+f"/{feat_filter}" if feat_filter != "filter" else iter
results = classification_report(y_test, y_test_pred, digits=4, output_dict=True)
with open(f"iter/{iter}/dt_res.json", "w") as f:
    json.dump(results, f)
print(classification_report(y_test, y_test_pred, digits=4))

def plot_feature_importance(model, feature_labels):
    # Extract feature importance from the Decision Tree
    feature_importances = model.named_steps['decisiontreeclassifier'].feature_importances_

    # Create a DataFrame for sorting
    importance_df = pd.DataFrame({
        "Feature": feature_labels,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False)

    importance_df.to_csv(f"iter/{iter}/tree_feat_importance.csv", index = False)

    # Plot feature importances as a horizontal bar chart
    plt.figure(figsize=(10, 20))
    plt.barh(importance_df["Feature"], importance_df["Importance"], color='skyblue')
    plt.xlabel("Feature Importance")
    plt.title("Feature Importance from Decision Tree Classifier")
    plt.gca().invert_yaxis()  # Highest importance at the top
    plt.tight_layout()
    plt.savefig(f"iter/{iter}/feat_importance_tree.png", dpi=300)
    plt.show()

def plot_decision_tree(model, feature_labels):
    # Extract the fitted Decision Tree model
    tree_model = model.named_steps['decisiontreeclassifier']

    # Plot the decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(tree_model, feature_names=feature_labels, filled=True, rounded=True, class_names=["Class 0", "Class 1"])
    plt.title("Decision Tree Structure")
    plt.savefig(f"iter/{iter}/decision_tree.png", dpi=300)
    plt.show()

def print_tree_rules(model, feature_labels, node=0, depth=0):
    """
    Recursively print the decision rules of a decision tree model.

    Parameters:
    - model: The trained DecisionTreeClassifier
    - feature_labels: List of feature names
    - node: The current node index in the tree
    - depth: The current depth in the tree (used for indentation)
    """
    tree = model.named_steps['decisiontreeclassifier'].tree_

    # Check if this is a leaf node
    if tree.children_left[node] == -1 and tree.children_right[node] == -1:
        print(f"{'|   ' * depth}Leaf node: class = {np.argmax(tree.value[node][0])}, "
              f"gini = {tree.impurity[node]:.3f}, samples = {tree.n_node_samples[node]}, "
              f"value = {tree.value[node][0]}")
    else:
        # Get the feature and threshold for the current node
        feature = feature_labels[tree.feature[node]]
        threshold = tree.threshold[node]

        # Print the decision rule at the current node
        print(f"{'|   ' * depth}{feature} <= {threshold:.3f} "
              f"(gini = {tree.impurity[node]:.3f}, samples = {tree.n_node_samples[node]}, "
              f"value = {tree.value[node][0]})")

        # Recursively print the left and right branches
        print_tree_rules(model, feature_labels, node=tree.children_left[node], depth=depth + 1)
        print_tree_rules(model, feature_labels, node=tree.children_right[node], depth=depth + 1)

# Plot feature importance and decision tree
plot_feature_importance(model, feature_labels)
plot_decision_tree(model, feature_labels)
# print_tree_rules(model, feature_labels)

r = export_text(model.named_steps['decisiontreeclassifier'], feature_names=feature_labels, show_weights=True)
with open(f"iter/{iter}/decision_tree_rules.txt", "w") as f:
    f.write(r)

    
