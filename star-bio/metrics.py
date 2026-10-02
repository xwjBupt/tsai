import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report


def classification_metrics(y, pred, labels):
    ids = np.arange(len(labels))
    _, recall, f1, _ = precision_recall_fscore_support(y, pred, labels=ids, zero_division=0)
    return {
        "acc": float(accuracy_score(y, pred)),
        "bacc": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "cm": confusion_matrix(y, pred, labels=ids).tolist(),
        "report": classification_report(y, pred, labels=ids, target_names=labels, output_dict=True, zero_division=0),
    }


@torch.inference_mode()
def evaluate(model, loader, device, labels):
    model.eval()
    ys, ps = [], []
    for x, y, _ in loader:
        pred = model(x.to(device)).argmax(1).cpu().numpy()
        ys.extend(y.tolist())
        ps.extend(pred.tolist())
    return classification_metrics(ys, ps, labels)
