import json
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

def save_confusion_matrix(preds, labels, model_name='', checkpoint=''):

    with open("data/label2id.json", "r") as f:
        label2id = json.load(f)
    id2label = {v:k for k, v in label2id.items()}
    classes = list(map(lambda x:id2label[x], sorted(set(labels))))

    print(labels)
    print('===')
    print(preds)
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f"Confusion Matrix - {model_name}")
    

    plt.savefig(f"{checkpoint}/confusion_matrix_{model_name}.png", bbox_inches='tight', dpi=300)
