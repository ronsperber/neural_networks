# import libraries
import numpy as np

def mse(y_pred, y_true):
    """
    Compute Mean Squared Error between predicted and true values.

    Parameters:
    y_pred (np.ndarray): Predicted values.
    y_true (np.ndarray): True values.

    Returns:
    float: Mean Squared Error.
    """
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    return np.mean((y_pred - y_true) ** 2)

def binary_accuracy(y_pred, y_true):
    """
    Compute Binary Accuracy between predicted and true binary values.

    Parameters:
    y_pred (np.ndarray): Predicted binary values (0 or 1).
    y_true (np.ndarray): True binary values (0 or 1).

    Returns:
    float: Binary Accuracy.
    """
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    y_pred_binary = np.round(y_pred)
    return np.mean(y_pred_binary == y_true)



def accuracy(y_pred, y_true):
    """
    Compute accuracy for binary or multi-class classification.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted values. Shape (batch_size,) for binary, (batch_size, num_classes) for multi-class.
    y_true : np.ndarray
        True values. Shape (batch_size,) integer labels.

    Returns
    -------
    float
        Accuracy score.
    """
    y_pred = np.asarray(y_pred)

    # binary case
    if y_pred.ndim == 1 or (y_pred.ndim == 2 and y_pred.shape[1] == 1):
        y_pred_binary = np.round(y_pred.ravel())
        return np.mean(y_pred_binary == y_true.ravel())
    
    # multi-class case
    else:
        y_pred_classes = np.argmax(y_pred, axis=1)
        return np.mean(y_pred_classes == y_true.ravel())

def categorical_accuracy(y_pred, y_true):
    """
    Compute Categorical Accuracy between predicted and true categorical values.

    Parameters:
    y_pred (np.ndarray): Predicted categorical values (one-hot encoded).
    y_true (np.ndarray): True categorical values (integer class numbers

    Returns:
    float: Categorical Accuracy.
    """
    y_pred_classes = np.argmax(y_pred, axis=1)

    return np.mean(y_pred_classes == y_true)

def mae(y_pred, y_true):
    """
    Compute Mean Absolute Error between predicted and true values.

    Parameters:
    y_pred (np.ndarray): Predicted values.
    y_true (np.ndarray): True values.

    Returns:
    float: Mean Absolute Error.
    """
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    return np.mean(np.abs(y_pred - y_true))

def precision(y_pred, y_true):
    """
    Compute Precision between predicted and true binary values.

    Parameters:
    y_pred (np.ndarray): Predicted binary values (0 or 1).
    y_true (np.ndarray): True binary values (0 or 1).

    Returns:
    float: Precision.
    """
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    y_pred_binary = np.round(y_pred)
    true_positives = np.sum((y_pred_binary == 1) & (y_true == 1))
    predicted_positives = np.sum(y_pred_binary == 1)
    
    if predicted_positives == 0:
        return 0.0
    
    return true_positives / predicted_positives

def recall(y_pred, y_true):
    """
    Compute Recall between predicted and true binary values.

    Parameters:
    y_pred (np.ndarray): Predicted binary values (0 or 1).
    y_true (np.ndarray): True binary values (0 or 1).

    Returns:
    float: Recall.
    """
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    y_pred_binary = np.round(y_pred)
    true_positives = np.sum((y_pred_binary == 1) & (y_true == 1))
    actual_positives = np.sum(y_true == 1)
    
    if actual_positives == 0:
        return 0.0
    
    return true_positives / actual_positives

def f1_score(y_pred, y_true):
    """
    Compute F1 Score between predicted and true binary values.

    Parameters:
    y_pred (np.ndarray): Predicted binary values (0 or 1).
    y_true (np.ndarray): True binary values (0 or 1).

    Returns:
    float: F1 Score.
    """
    prec = precision(y_pred, y_true)
    rec = recall(y_pred, y_true)
    
    if (prec + rec) == 0:
        return 0.0
    
    return 2 * (prec * rec) / (prec + rec)

def r2_score(y_pred, y_true):
    """
    Compute R-squared (Coefficient of Determination) between predicted and true values.

    Parameters:
    y_pred (np.ndarray): Predicted values.
    y_true (np.ndarray): True values.

    Returns:
    float: R-squared value.
    """
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)

def specificity(y_pred, y_true):
    """
    Compute Specificity between predicted and true binary values.

    Parameters:
    y_pred (np.ndarray): Predicted binary values (0 or 1).
    y_true (np.ndarray): True binary values (0 or 1).

    Returns:
    float: Specificity.
    """
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    y_pred_binary = np.round(y_pred)
    true_negatives = np.sum((y_pred_binary == 0) & (y_true == 0))
    actual_negatives = np.sum(y_true == 0)
    
    if actual_negatives == 0:
        return 0.0
    
    return true_negatives / actual_negatives

def balanced_accuracy(y_pred, y_true):
    """
    Compute Balanced Accuracy between predicted and true binary values.

    Parameters:
    y_pred (np.ndarray): Predicted binary values (0 or 1).
    y_true (np.ndarray): True binary values (0 or 1).

    Returns:
    float: Balanced Accuracy.
    """
    rec = recall(y_pred, y_true)
    spec = specificity(y_pred, y_true)
    
    return (rec + spec) / 2

metrics_dict = {
    "mse": mse,
    "accuracy": accuracy,
    "binary_accuracy": binary_accuracy,
    "categorical_accuracy": categorical_accuracy,
    "mae": mae,
    "precision": precision,
    "recall": recall,
    "f1_score": f1_score,
    "r2_score": r2_score,
    "specificity": specificity,
    "balanced_accuracy": balanced_accuracy
}