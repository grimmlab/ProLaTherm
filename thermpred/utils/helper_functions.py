import os
import inspect
import importlib
import torch
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import pathlib
import h5py

import thermpred.model


def get_list_of_implemented_models() -> list:
    """
    Create a list of all implemented models based on files existing in 'model' subdirectory of the repository.
    """
    # Assumption: naming of python source file is the same as the model name specified by the user
    if os.path.exists('../model'):
        model_src_files = os.listdir('../model')
    elif os.path.exists('model'):
        model_src_files = os.listdir('model')
    elif os.path.exists('thermpred/model'):
        model_src_files = os.listdir('thermpred/model')
    else:
        model_src_files = [model_file + '.py' for model_file in thermpred.model.__all__]
    model_src_files = [file for file in model_src_files if file[0] != '_']
    return [model[:-3] for model in model_src_files]


def test_likely_categorical(vector_to_test: list, abs_unique_threshold: int = 20) -> bool:
    """
    Test whether a vector is most likely categorical.
    Simple heuristics: checking if the number of unique values exceeds a specified threshold

    :param vector_to_test: vector that is tested if it is most likely categorical
    :param abs_unique_threshold: threshold of unique values' ratio to declare vector categorical

    :return: True if the vector is most likely categorical, False otherwise
    """
    number_unique_values = np.unique(vector_to_test).shape[0]
    return number_unique_values <= abs_unique_threshold


def get_mapping_name_to_class() -> dict:
    """
    Get a mapping from model name (naming in package model without .py) to class name.

    :return: dictionary with mapping model name to class name
    """
    if os.path.exists('../model'):
        files = os.listdir('../model')
    elif os.path.exists('model'):
        files = os.listdir('model')
    elif os.path.exists('thermpred/model'):
        files = os.listdir('thermpred/model')
    else:
        files = [model_file + '.py' for model_file in thermpred.model.__all__]
    modules_mapped = {}
    for file in files:
        if file not in ['__init__.py', '__pycache__']:
            if file[-3:] != '.py':
                continue

            file_name = file[:-3]
            module_name = 'thermpred.model.' + file_name
            for name, cls in inspect.getmembers(importlib.import_module(module_name), inspect.isclass):
                if cls.__module__ == module_name:
                    modules_mapped[file_name] = cls
    return modules_mapped


def set_all_seeds(seed: int = 42):
    """
    Set all seeds of libs with a specific function for reproducibility of results

    :param seed: seed to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tf.random.set_seed(seed)


def get_subpath_for_datasplit(datasplit: str, datasplit_params: list) -> str:
    """
    Construct the subpath according to the datasplit.

    Datasplit parameters:

    - nested-cv: [n_outerfolds, n_innerfolds]
    - cv-test: [n_innerfolds, test_set_size_percentage]
    - train-val-test: [val_set_size_percentage, test_set_size_percentage]

    :param datasplit: datasplit to retrieve
    :param datasplit_params: parameters to use for the specific datasplit

    :return: string with the subpath
    """
    # construct subpath due to the specified datasplit
    if datasplit == 'train-val-test':
        datasplit_string = f'({100 - datasplit_params[0]}-{datasplit_params[0]})-{datasplit_params[1]}'
    elif datasplit == 'cv-test':
        datasplit_string = f'{datasplit_params[0]}-{datasplit_params[1]}'
    elif datasplit == 'nested-cv':
        datasplit_string = f'{datasplit_params[0]}-{datasplit_params[1]}'
    return datasplit_string


def save_model_overview_dict(model_overview: dict, save_path: str):
    """
    Structure and save results of a whole optimization run for multiple models in one csv file

    :param model_overview: dictionary with results overview
    :param save_path: filepath for saving the results overview file
    """
    results_overview = pd.DataFrame()
    for model_name, fold_dicts in model_overview.items():
        result_dicts = {}
        result_dicts_std = {}
        runtime_dicts = {}
        runtime_dicts_std = {}
        for fold_name, fold_info in fold_dicts.items():
            for result_name, result_info in fold_info.items():
                results_overview.at[fold_name, model_name + '___' + result_name] = [result_info]
                if 'eval_metric' in result_name:
                    for metric_name, metric_result in result_info.items():
                        if 'list' in metric_name:
                            continue
                        if metric_name not in result_dicts.keys():
                            result_dicts[metric_name] = []
                        result_dicts[metric_name].append(metric_result)
                if 'runtime' in result_name:
                    for metric_name, metric_result in result_info.items():
                        if metric_name not in runtime_dicts.keys():
                            runtime_dicts[metric_name] = []
                        runtime_dicts[metric_name].append(metric_result)
        for metric_name, results in result_dicts.items():
            if 'list' in metric_name:
                continue
            else:
                result_dicts[metric_name] = np.mean(results)
                result_dicts_std[metric_name] = np.std(results)
        for metric_name, results in runtime_dicts.items():
            runtime_dicts[metric_name] = np.mean(results)
            runtime_dicts_std[metric_name] = np.std(results)
        if 'nested' in str(save_path):
            results_overview.at['mean_over_all_folds', model_name + '___' + 'eval_metrics'] = [result_dicts]
            results_overview.at['std_over_all_folds', model_name + '___' + 'eval_metrics'] = [result_dicts_std]
            results_overview.at['mean_over_all_folds', model_name + '___' + 'runtime_metrics'] = [runtime_dicts]
            results_overview.at['std_over_all_folds', model_name + '___' + 'runtime_metrics'] = [runtime_dicts_std]
    results_overview.to_csv(save_path)


def sort_models_by_featureset(models_list: list) -> list:
    """
    Sort models by the encoding that will be used

    :param models_list: unsorted list of models

    :return: list of models sorted by encoding
    """
    featuresets = [get_mapping_name_to_class()[model_name].featureset for model_name in models_list]
    sorted_models_list = [el[0] for el in sorted(zip(models_list, featuresets), key=lambda x: x[1])]
    return sorted_models_list


def get_all_subdirectories_non_recursive(path: pathlib.Path) -> list:
    """
    Get all non-recursive subdirectories of path

    :param path: path to search

    :return: list with all non-recursive subdirs
    """
    return list(filter(lambda x: x.is_dir(), path.iterdir()))


def get_all_files(path: pathlib.Path) -> list:
    """
    Get all non-recursive files of path

    :param path: path to search

    :return: list with all non-recursive files
    """
    return list(filter(lambda x: x.is_file(), path.iterdir()))


def get_all_files_with_suffix(path: pathlib.Path, suffix: str) -> list:
    """
    Get all non-recursive files of path

    :param path: path to search
    :param suffix: suffix to search

    :return: list with all non-recursive files
    """
    return list(filter(lambda x: x.suffix == suffix, path.iterdir()))


def get_datasplit_config_info_for_resultfolder(resultfolder: str) -> tuple:
    """
    Get all datasplit info for a result folder

    :param resultfolder: path to retrieve info

    :return: datasplit info with datasplit, n_outerfolds, n_innerfolds, val_set_size_percentage, test_set_size_percentage, maf_percentage
    """
    maf_perc = int(resultfolder.split('_')[2][3:])
    datasplit = resultfolder.split('_')[0]
    n_outerfolds = 1
    n_innerfolds = 5
    test_set_size_percentage = 20
    val_set_size_percentage = 20
    if datasplit == 'nested-cv':
        n_outerfolds = int(resultfolder.split('_')[1].split('-')[0])
        n_innerfolds = int(resultfolder.split('_')[1].split('-')[1])
    elif datasplit == 'cv-test':
        n_innerfolds = int(resultfolder.split('_')[1].split('-')[0])
        test_set_size_percentage = int(resultfolder.split('_')[1].split('-')[1])
    else:
        val_set_size_percentage = int(resultfolder.split('_')[1].split('-')[1][:-1])
        test_set_size_percentage = int(resultfolder.split('_')[1].split('-')[-1])
    return datasplit, n_outerfolds, n_innerfolds, val_set_size_percentage, test_set_size_percentage, maf_perc


def load_embeddings_for_prot_ids(prot_ids: list, embedding_file: pathlib.Path, max_len: int,
                                 pretrained='prot_t5_xl_uniref50'):
    X = np.zeros((len(prot_ids), max_len, 1024), dtype=np.float32)
    with h5py.File(embedding_file, 'r') as f:
        for index in range(len(prot_ids)):
            sample_id = prot_ids[index]
            embedding = f['/' + pretrained + '/' + sample_id][:].astype(np.float32)
            embedding = embedding[:max_len, :] if embedding.shape[0] > max_len else embedding
            matrix = np.zeros((max_len, 1024), dtype=np.float32)
            matrix[:embedding.shape[0], :] = embedding
            X[index] = matrix
    return X
