"""
Metrics computation module.
Calculates all paper-specified evaluation metrics + Stage 3 assignment QWK & per-class metrics:

Paper Metrics:
1. Accuracy
2. Precision (macro)
3. Recall / Sensitivity (macro)
4. F1 Score (macro)
5. Jaccard Index (JAC)
6. Dice Coefficient (DIC)
7. Matthews Correlation Coefficient (MCC)
8. Specificity (macro)
9. Area Under Curve (AUC)

Stage 3 Metric (Assignment requirement, NOT from paper):
10. Quadratic Weighted Kappa (QWK)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix, cohen_kappa_score
)

CLASS_NAMES = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]

def compute_all_metrics(y_true_onehot, y_pred_probs):
    """
    Compute overall and per-class metrics.
    
    y_true_onehot: (N, 7) one-hot encoded true labels
    y_pred_probs: (N, 7) predicted probability distribution
    """
    y_true = np.argmax(y_true_onehot, axis=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    num_classes = y_true_onehot.shape[1]

    # Global accuracy
    acc = accuracy_score(y_true, y_pred) * 100.0

    # Macro-averaged metrics
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100.0
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100.0
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100.0
    mcc = matthews_corrcoef(y_true, y_pred) * 100.0

    # Stage 3 assignment metric: Quadratic Weighted Kappa (QWK)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')

    # Multi-class AUC (One-vs-Rest)
    try:
        auc = roc_auc_score(y_true_onehot, y_pred_probs, multi_class='ovr', average='macro') * 100.0
    except Exception:
        auc = 0.0

    # Per-class TP, TN, FP, FN calculation
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    total_samples = np.sum(cm)

    specificities = []
    jaccard_scores = []
    dice_scores = []
    per_class_recalls = []
    per_class_precisions = []

    per_class_rows = []

    for i in range(num_classes):
        cls_name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class_{i}"
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = total_samples - (tp + fp + fn)

        cls_prec = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
        cls_rec = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
        cls_spec = (tn / (tn + fp)) * 100.0 if (tn + fp) > 0 else 0.0
        cls_f1 = (2 * cls_prec * cls_rec / (cls_prec + cls_rec)) if (cls_prec + cls_rec) > 0 else 0.0
        cls_jac = (tp / (tp + fn + fp)) * 100.0 if (tp + fn + fp) > 0 else 0.0
        cls_dic = (2 * tp / (2 * tp + fp + fn)) * 100.0 if (2 * tp + fp + fn) > 0 else 0.0

        specificities.append(cls_spec)
        jaccard_scores.append(cls_jac)
        dice_scores.append(cls_dic)
        per_class_recalls.append(cls_rec)
        per_class_precisions.append(cls_prec)

        per_class_rows.append({
            'Class': cls_name,
            'TP': int(tp),
            'FP': int(fp),
            'TN': int(tn),
            'FN': int(fn),
            'Precision (%)': round(cls_prec, 2),
            'Recall / Sensitivity (%)': round(cls_rec, 2),
            'Specificity (%)': round(cls_spec, 2),
            'F1 Score (%)': round(cls_f1, 2),
            'Jaccard (%)': round(cls_jac, 2),
            'Dice (%)': round(cls_dic, 2)
        })

    macro_specificity = float(np.mean(specificities))
    macro_jaccard = float(np.mean(jaccard_scores))
    macro_dice = float(np.mean(dice_scores))

    metrics_dict = {
        'Accuracy': float(acc),
        'Precision': float(precision),
        'F1 Score': float(f1),
        'Sensitivity (Recall)': float(recall),
        'Overall JAC score': float(macro_jaccard),
        'Overall DIC Score': float(macro_dice),
        'Overall MCC Score': float(mcc),
        'Specificity': float(macro_specificity),
        'AUC': float(auc),
        'Quadratic Weighted Kappa (QWK)': float(qwk)
    }

    per_class_df = pd.DataFrame(per_class_rows)

    return metrics_dict, cm, per_class_df
