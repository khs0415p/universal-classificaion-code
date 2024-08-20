import json
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def apply_format(row, df_ratio):
    output = []
    
    for v, r in zip(row, df_ratio.loc[row.name]):
        r *= 100
        output.append(f"{v} ({r:.1f}%)")
    
    return output

def save_confusion_matrix(labels, confusion_matrix, model_name='', checkpoint=''):
    with open("data/label2id.json", "r") as f:
        label2id = json.load(f)
    id2label = {v:k for k, v in label2id.items()}
    classes = list(map(lambda x:id2label[x], sorted(set(labels))))

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f"Confusion Matrix - {model_name}")

    plt.savefig(f"{checkpoint}/confusion_matrix_{model_name}.png", bbox_inches='tight', dpi=300)

    # confusion matrix
    confusion_df = pd.DataFrame(confusion_matrix, columns=classes, index=classes)

    df_sum = confusion_df.sum(axis=1)
    df_ratio = confusion_df.div(df_sum, axis=1)

    df_formatted = confusion_df.apply(apply_format, df_ratio=df_ratio, axis=1)
    df_formatted = pd.DataFrame(df_formatted.tolist(), columns=confusion_df.columns, index=confusion_df.index)

    df_formatted.to_csv(f"{checkpoint}/confusion_matrix_{model_name}.csv")


def get_metrics(preds, labels):

    return f1_score(labels, preds, average="weighted"), accuracy_score(labels, preds), confusion_matrix(labels, preds)