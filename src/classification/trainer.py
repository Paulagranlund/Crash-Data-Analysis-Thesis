"""Class weights, weighted loss, and metrics for imbalanced crash classes.

The weighted loss is supplied to the HuggingFace Trainer through its
``compute_loss_func`` argument, so no Trainer subclass is required.
"""
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score, f1_score


def compute_class_weights(df_train, label_col, num_labels):
    """Inverse-frequency class weights, normalised to sum to num_labels.

    Rare classes receive larger weights so they contribute more to the loss.
    Returns a one-dimensional numpy array of length num_labels, ordered by
    label id.
    """
    class_counts = df_train[label_col].value_counts().sort_index().values
    weights = 1.0 / (class_counts + 1e-9)
    weights = weights / weights.sum() * num_labels
    return weights


def make_weighted_loss(class_weights):
    """Return a ``compute_loss_func`` that applies class weights.

    The returned function matches the signature the Trainer expects and moves
    the weights onto the logits' device at call time, so it works on CPU or
    GPU without extra setup. Common classes are penalised less and rare classes
    more, counteracting the imbalance across crash-situation classes.
    """
    weights = torch.tensor(class_weights, dtype=torch.float)

    def weighted_loss(outputs, labels, num_items_in_batch=None):
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        loss_fct = CrossEntropyLoss(weight=weights.to(logits.device))
        return loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

    return weighted_loss


def compute_metrics(eval_pred):
    """Accuracy and macro-F1 for the HuggingFace Trainer.

    Macro-F1 weighs every class equally, which suits the imbalanced label set.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }
