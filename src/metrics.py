"""
Metrics computation module.
Calculates all 9 paper-specified evaluation metrics:
1. Accuracy
2. Precision
3. Sensitivity (Recall)
4. F1 Score
5. Jaccard Index (JAC)
6. Dice Coefficient (DIC)
7. Matthews Correlation Coefficient (MCC)
8. Specificity
9. Area Under Curve (AUC)
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix
)

def compute_all_metrics(y_true_onehot, y_pred_probs):
    """
    Compute comprehensive metrics array matching paper Section III.D definitions.
    
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

    # Multi-class AUC (One-vs-Rest)
    try:
        auc = roc_auc_score(y_true_onehot, y_pred_probs, multi_class='ovr', average='macro') * 100.0
    except Exception:
        auc = 0.0

    # Per-class TP, TN, FP, FN calculation for Specificity, JAC, DIC
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    specificities = []
    jaccard_scores = []
    dice_scores = []

    total_samples = np.sum(cm)

    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = total_samples - (tp + fp + fn)

        # Specificity = TN / (TN + FP)
        spec = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        specificities.append(spec)

        # JAC = TP / (TP + FN + FP)
        jac = (tp / (tp + fn + fp)) if (tp + fn + fp) > 0 else 0.0
        jaccard_scores.append(jac)

        # DIC = (2 * TP) / (2 * TP + FP + FN)
        dic = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
        dice_scores.append(dic)

    macro_specificity = np.mean(specificities) * 100.0
    macro_jaccard = np.mean(jaccard_scores) * 100.0
    macro_dice = np.mean(dice_scores) * 100.0

    metrics_dict = {
        'Accuracy': float(acc),
        'Precision': float(precision),
        'F1 Score': float(f1),
        'Sensitivity (Recall)': float(recall),
        'Overall JAC score': float(macro_jaccard),
        'Overall DIC Score': float(macro_dice),
        'Overall MCC Score': float(mcc),
        'Specificity': float(macro_specificity),
        'AUC': float(auc)
    }

    return metrics_dict, cm
