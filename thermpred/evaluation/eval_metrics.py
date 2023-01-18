import sklearn
import numpy as np


def get_evaluation_report(y_pred: np.array, y_true: np.array, y_score: np.array, task: str, prefix: str = '') -> dict:
    """
    Get values for common evaluation metrics

    :param y_pred: predicted values
    :param y_true: true values
    :param y_score: scores for predict value
    :param task: ML task to solve
    :param prefix: prefix to be added to the key if multiple eval metrics are collected

    :return: dictionary with common metrics
    """
    if len(y_pred) == (len(y_true)-1):
        print('y_pred has one element less than y_true (e.g. due to batch size config) -> dropped last element')
        y_true = y_true[:-1]
    if task == 'classification':
        average = 'micro' if len(np.unique(y_true)) > 2 else 'binary'
        roc_fpr, roc_tpr, roc_thr = sklearn.metrics.roc_curve(y_true=y_true, y_score=y_score)
        eval_report_dict = {
            prefix + 'accuracy': sklearn.metrics.accuracy_score(y_true=y_true, y_pred=y_pred),
            prefix + 'f1_score': sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred, average=average),
            prefix + 'precision': sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred,
                                                                  zero_division=0, average=average),
            prefix + 'recall': sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred,
                                                            zero_division=0, average=average),
            prefix + 'mcc': sklearn.metrics.matthews_corrcoef(y_true=y_true, y_pred=y_pred),
            prefix + 'roc_list_fpr': roc_fpr,
            prefix + 'roc_list_tpr': roc_tpr,
            prefix + 'roc_list_threshold': roc_thr,
            prefix + 'roc_auc': sklearn.metrics.roc_auc_score(y_true=y_true, y_score=y_score)
        }
    else:
        eval_report_dict = {
            prefix + 'mse': sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred),
            prefix + 'rmse': sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False),
            prefix + 'r2_score': sklearn.metrics.r2_score(y_true=y_true, y_pred=y_pred),
            prefix + 'explained_variance': sklearn.metrics.explained_variance_score(y_true=y_true, y_pred=y_pred)
        }
    return eval_report_dict
