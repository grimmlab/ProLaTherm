from prolatherm.prolatherm import ProLaTherm
from thermpred.preprocess import base_dataset
from thermpred.evaluation import eval_metrics
import pathlib
import numpy as np
import torch
import pandas as pd


if __name__ == '__main__':
    dataset = base_dataset.Dataset(
        data_dir=pathlib.Path("/bit_storage/Projects/ProteinThermPred/final_datasets/new_data/"),
        dataset_name='ProtThermPred_exp2.csv', datasplit='cv-test', n_outerfolds=1, n_innerfolds=5,
        test_set_size_percentage=20, val_set_size_percentage=20, featureset='sequence'
    )
    raw_data = pd.read_csv("/bit_storage/Projects/ProteinThermPred/final_datasets/new_data/ProtThermPred_exp2.csv")
    pred_model = ProLaTherm()
    test_indices = dataset.datasplit_indices['outerfold_0']['test']
    train_indices = ~np.isin(np.arange(len(dataset.X_full)), test_indices)
    X_test = torch.tensor(dataset.X_full[test_indices])
    y_test = dataset.y_full[test_indices]
    y_train = dataset.y_full[train_indices]
    seqs_test = list(raw_data.iloc[test_indices, :]['seq_peptide'])[0:10]
    seqs_train = list(raw_data.iloc[train_indices, :]['seq_peptide'])
    preds, scores = pred_model.predict(seqs_test)
    print(eval_metrics.get_evaluation_report(
        y_pred=preds.cpu().detach().numpy().reshape(-1, 1), y_true=y_test,
        y_score=scores.cpu().detach().numpy().reshape(-1, 1), task='classification')
    )
