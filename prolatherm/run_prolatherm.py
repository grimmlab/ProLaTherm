import sklearn.metrics
import numpy as np
import pandas as pd

import prolatherm


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
        #roc_fpr, roc_tpr, roc_thr = sklearn.metrics.roc_curve(y_true=y_true, y_score=y_score)
        eval_report_dict = {
            prefix + 'accuracy': sklearn.metrics.accuracy_score(y_true=y_true, y_pred=y_pred),
            prefix + 'f1_score': sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred, average=average),
            prefix + 'precision': sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred,
                                                                  zero_division=0, average=average),
            prefix + 'recall': sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred,
                                                            zero_division=0, average=average),
            prefix + 'mcc': sklearn.metrics.matthews_corrcoef(y_true=y_true, y_pred=y_pred)
            #prefix + 'roc_list_fpr': roc_fpr,
            #prefix + 'roc_list_tpr': roc_tpr,
            #prefix + 'roc_list_threshold': roc_thr,
            #prefix + 'roc_auc': sklearn.metrics.roc_auc_score(y_true=y_true, y_score=y_score)
        }
    else:
        eval_report_dict = {
            prefix + 'mse': sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred),
            prefix + 'rmse': sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False),
            prefix + 'r2_score': sklearn.metrics.r2_score(y_true=y_true, y_pred=y_pred),
            prefix + 'explained_variance': sklearn.metrics.explained_variance_score(y_true=y_true, y_pred=y_pred)
        }
    return eval_report_dict


if __name__ == '__main__':
    raw_data = pd.read_csv("/myhome/ProLaTherm/data/datasets_w_datasplits/ProtThermPred_speciesspecific.csv")
    unique_data = raw_data.iloc[4797:4820]
    prot_ids_test = unique_data["meta_protein_id"]
    seqs_test = unique_data['seq_peptide']
    y_test = unique_data['label_binary']
    pred_model = prolatherm.ProLaTherm()

    start = 0
    step_size = 10
    num_samples = unique_data.shape[0]
    preds = []
    scores = []
    for i in range(step_size, num_samples, step_size):
        end = i if i + step_size < num_samples else num_samples
        print(start)
        print(end)
        # prot_ids = prot_ids_test[start:end]
        seq = seqs_test[start:end]
        print("Running prediction")
        print(seq)
        pred, score = pred_model.predict(seq)
        print(pred)
        print(score)
        preds.extend(pred.flatten().tolist())
        scores.extend(score.flatten().tolist())
        start = end
    print(get_evaluation_report(y_pred=np.array(preds), y_score=np.array(scores), y_true=np.array(y_test), task="classification"))

    # fasta file laden, das als param übergeben werden kann
    # daten batchen und predictions machen
    # csv abspeichern
