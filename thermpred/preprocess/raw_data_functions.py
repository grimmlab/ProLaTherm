import numpy as np
import pandas as pd
import h5py
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from itertools import groupby
import iFeatureOmegaCLI

from ..utils import helper_functions, check_functions


def prepare_data_files(data_dir: pathlib.Path, dataset_name: str, fasta_file_thermo: str, fasta_file_non_thermo: str,
                       name_new_dataset: str, datasplit: str, n_outerfolds: int, n_innerfolds: int,
                       test_set_size_percentage: int, val_set_size_percentage: int):
    """
    Prepare all data files for a common format in a csv file.

    First check if csv file is provided via dataset_name and is in a valid format.

    If no csv file is provided via dataset_name, unify the two fasta files to create a new .csv file named after name_new_dataset.

    :param data_dir: data directory where the phenotype and genotype matrix are stored
    :param dataset_name: name of the dataset to use
    :param fasta_file_thermo: name of the fasta file containing thermophilic proteins
    :param fasta_file_non_thermo: name of the fasta file containing non-thermophilic proteins
    :param name_new_dataset: name of the new dataset based on the two provided fasta files
    :param datasplit: datasplit to use. Options are: nested-cv, cv-test, train-val-test
    :param n_outerfolds: number of outerfolds relevant for nested-cv
    :param n_innerfolds: number of folds relevant for nested-cv and cv-test
    :param test_set_size_percentage: size of the test set relevant for cv-test and train-val-test
    :param val_set_size_percentage: size of the validation set relevant for train-val-test
    """
    if dataset_name is not None:
        print("Check if .csv file provided via dataset is in valid format")
        check_dataset_format(data_dir=data_dir, dataset_name=dataset_name)
        print('Data format valid')
    else:
        print("Create dataset based on two provided fasta files")
        create_dataset(
            data_dir=data_dir, fasta_file_thermo=fasta_file_thermo, fasta_file_non_thermo=fasta_file_non_thermo,
            name_new_dataset=name_new_dataset
        )
        dataset_name = name_new_dataset + '.csv' if '.csv' not in name_new_dataset else name_new_dataset

    if data_dir.joinpath('index_file_' + dataset_name.split('.')[0] + '.h5').is_file():
        print('Index file already exists.Will append required filters and data splits now.')
        append_index_file(
            data_dir=data_dir, dataset_name=dataset_name, datasplit=datasplit, n_outerfolds=n_outerfolds,
            n_innerfolds=n_innerfolds, test_set_size_percentage=test_set_size_percentage,
            val_set_size_percentage=val_set_size_percentage
        )
        print('Done checking data files. All required datasets are available.')
    else:
        print('Index file does not exist. Will create new index file.')
        create_index_file(
            data_dir=data_dir, dataset_name=dataset_name, datasplit=datasplit, n_outerfolds=n_outerfolds,
            n_innerfolds=n_innerfolds, test_set_size_percentage=test_set_size_percentage,
            val_set_size_percentage=val_set_size_percentage
        )
        print('Done checking data files. All required datasets are available.')


def check_dataset_format(data_dir: pathlib.Path, dataset_name: str):
    """
    Check format of the specified .csv dataset file

    :param data_dir: data directory containing the dataset .csv file
    :param dataset_name: name of the dataset .csv file
    """
    # check format (feat_, seq_, label_, .... vorhanden in .csv, mehr nicht)
    dataset_raw = pd.read_csv(data_dir.joinpath(dataset_name))
    if 'meta_protein_id' not in dataset_raw.columns:
        raise Exception('meta_protein_id not in provided dataset')
    if 'label_binary' not in dataset_raw.columns:
        raise Exception('label_binary not in provided dataset')
    if 'seq_peptide' not in dataset_raw.columns:
        raise Exception('seq_peptide not in provided dataset')
    if not any('feat' in col for col in dataset_raw.columns):
        raise Exception('No column with prefix feat in whole dataset')
    if not check_functions.check_exist_files([data_dir.joinpath('embeddings_' + dataset_name.split('.')[0] + '.h5')]):
        print('.h5 file containing embeddings does not exist, so you cannot use such models!')
    # TODO: prüfen wenn 3d daten auch drin
    #if not any('struct' in col for col in dataset_raw.columns):
    #    raise Exception('No column with prefix struct in whole dataset')


def create_dataset(data_dir: pathlib.Path, fasta_file_thermo: str, fasta_file_non_thermo: str, name_new_dataset: str):
    """
    Create new dataset called name_new_dataset based on two provided fasta files

    :param data_dir: data directory containing the dataset .csv file
    :param fasta_file_thermo: fasta file containing the thermophilic proteins
    :param fasta_file_non_thermo: fasta file containing the non-thermophilic proteins
    :param name_new_dataset: name of the new dataset
    """
    # load fasta files
    seq_thermo, ids_thermo = read_fasta(filepath=data_dir.joinpath(fasta_file_thermo))
    seq_non_thermo, ids_non_thermo = read_fasta(filepath=data_dir.joinpath(fasta_file_non_thermo))
    # delete ambiguous aa if still present
    seq_thermo, ids_thermo = remove_ambiguous_aa(sequences=seq_thermo, protein_ids=ids_thermo)
    seq_non_thermo, ids_non_thermo = remove_ambiguous_aa(sequences=seq_non_thermo, protein_ids=ids_non_thermo)
    # create dataframe with unified data
    thermo_data = pd.DataFrame()
    non_thermo_data = pd.DataFrame()
    for info in zip(
            [thermo_data, non_thermo_data],
            [ids_thermo, ids_non_thermo],
            [fasta_file_thermo, fasta_file_non_thermo],
            ['thermo', 'meso'],
            [1, 0],
            [seq_thermo, seq_non_thermo]):
        dataframe = info[0]
        dataframe['meta_protein_id'] = info[1]
        dataframe['meta_file_name'] = info[2]
        dataframe['label_description'] = info[3]
        dataframe['label_binary'] = info[4]
        dataframe['seq_peptide'] = info[5]
    unified_data = pd.concat([thermo_data, non_thermo_data]).reset_index(drop=True)
    print('--- filter duplicates ---')
    duplicates = unified_data[unified_data.duplicated(subset=['meta_protein_id'])]['meta_protein_id']
    if duplicates.shape[0] != 0:
        print('Found the following duplicates. Will drop duplicates.')
        print(duplicates)
        unified_data = unified_data.drop_duplicates(subset=['meta_protein_id'], ignore_index=True)
    # add all features to dataframe
    unified_data_w_features = add_features(
        raw_data=unified_data,
        fasta_file_paths=[data_dir.joinpath(fasta_file_thermo), data_dir.joinpath(fasta_file_non_thermo)],
        features_to_add=["Basic", "AAC", "CTDC", "CTDT", "CTDD", "PAAC", "DPC type 1"]
    )
    unified_data_w_features = unified_data_w_features.drop_duplicates(subset=['meta_protein_id'], ignore_index=True)
    full_path_new_dataset = data_dir.joinpath(name_new_dataset) if '.csv' in name_new_dataset \
        else data_dir.joinpath(name_new_dataset + '.csv')
    unified_data_w_features.to_csv(full_path_new_dataset, index=0)
    print('new unified dataset saved at ' + str(full_path_new_dataset))
    print('In case you want to use models using embeddings from pretrained models, '
          'do not forget to generate embeddings!')


def append_index_file(data_dir: pathlib.Path, dataset_name: str, datasplit: str, n_outerfolds: int, n_innerfolds: int,
                      test_set_size_percentage: int, val_set_size_percentage: int):
    """
    Check index file, described in create_index_file(), and append datasets if necessary

    :param data_dir: data directory where the phenotype and genotype matrix are stored
    :param dataset_name: name of dataset provided as .csv file to use
    :param datasplit: datasplit to use. Options are: nested-cv, cv-test, train-val-test
    :param n_outerfolds: number of outerfolds relevant for nested-cv
    :param n_innerfolds: number of folds relevant for nested-cv and cv-test
    :param test_set_size_percentage: size of the test set relevant for cv-test and train-val-test
    :param val_set_size_percentage: size of the validation set relevant for train-val-test
    """
    with h5py.File(data_dir.joinpath('index_file_' + dataset_name.split('.')[0] + '.h5'), 'a') as f:
        # check if group 'maf_filter' is available and if user input maf is available, if not: create group/dataset
        # check if group datasplit and all user inputs concerning datasplits are available, if not: create all
        if datasplit == 'nested-cv':
            subpath = helper_functions.get_subpath_for_datasplit(datasplit=datasplit,
                                                                 datasplit_params=[n_outerfolds, n_innerfolds])
            if 'datasplits' not in f or ('datasplits' in f and 'nested-cv' not in f['datasplits']) or \
                    ('datasplits' in f and 'nested-cv' in f['datasplits'] and
                     f'{subpath}' not in f['datasplits/nested-cv']):
                nest = f.create_group(f'datasplits/nested-cv/{subpath}')
                index_dict = check_train_test_splits(y=f['full_data/y'], datasplit='nested-cv',
                                                     datasplit_params=[n_outerfolds, n_innerfolds])
                for outer in range(n_outerfolds):
                    o = nest.create_group(f'outerfold_{outer}')
                    o.create_dataset('test', data=index_dict[f'outerfold_{outer}_test'], chunks=True,
                                     compression="gzip")
                    for inner in range(n_innerfolds):
                        i = o.create_group(f'innerfold_{inner}')
                        i.create_dataset('train', data=index_dict[f'outerfold_{outer}'][f'fold_{inner}_train'],
                                         chunks=True,
                                         compression="gzip")
                        i.create_dataset('val', data=index_dict[f'outerfold_{outer}'][f'fold_{inner}_test'],
                                         chunks=True,
                                         compression="gzip")
        elif datasplit == 'cv-test':
            subpath = helper_functions.get_subpath_for_datasplit(
                datasplit=datasplit, datasplit_params=[n_innerfolds, test_set_size_percentage]
            )
            if 'datasplits' not in f or ('datasplits' in f and 'cv-test' not in f['datasplits']) or \
                    ('datasplits' in f and 'cv-test' in f['datasplits'] and
                     f'{subpath}' not in f['datasplits/cv-test']):
                cv = f.create_group(f'datasplits/cv-test/{subpath}')
                index_dict, test = check_train_test_splits(y=f['full_data/y'], datasplit='cv-test',
                                                           datasplit_params=[n_innerfolds, test_set_size_percentage])
                o = cv.create_group('outerfold_0')
                o.create_dataset('test', data=test, chunks=True, compression="gzip")
                for fold in range(n_innerfolds):
                    i = o.create_group(f'innerfold_{fold}')
                    i.create_dataset('train', data=index_dict[f'fold_{fold}_train'], chunks=True, compression="gzip")
                    i.create_dataset('val', data=index_dict[f'fold_{fold}_test'], chunks=True, compression="gzip")
        elif datasplit == 'train-val-test':
            subpath = helper_functions.get_subpath_for_datasplit(datasplit=datasplit,
                                                                 datasplit_params=[val_set_size_percentage,
                                                                                   test_set_size_percentage])
            if 'datasplits' not in f or ('datasplits' in f and 'train-val-test' not in f['datasplits']) or \
                    ('datasplits' in f and 'train-val-test' in f['datasplits'] and
                     f'{subpath}' not in f['datasplits/train-val-test']):
                tvt = f.create_group(f'datasplits/train-val-test/{subpath}')
                train, val, test = check_train_test_splits(y=f['full_data/y'], datasplit='train-val-test',
                                                           datasplit_params=[val_set_size_percentage,
                                                                             test_set_size_percentage])
                o = tvt.create_group('outerfold_0')
                o.create_dataset('test', data=test, chunks=True, compression="gzip")
                i = o.create_group('innerfold_0')
                i.create_dataset('train', data=train, chunks=True, compression="gzip")
                i.create_dataset('val', data=val, chunks=True, compression="gzip")


def create_index_file(data_dir: pathlib.Path, dataset_name: str, datasplit: str, n_outerfolds: int, n_innerfolds: int,
                      test_set_size_percentage: int, val_set_size_percentage: int):
    """
    Create the .h5 index file containing the datasplits.
    It will be created using standard values additionally to user inputs.

    Unified format of .h5 file containing the data splits:

    .. code-block:: python

            'full_data': {
                    'all_sample_ids': sample ids of whole dataset,
                    'y': all y values
                    }
            'datasplits': {
                    'nested_cv': {
                            '#outerfolds-#innerfolds': {
                                    'outerfold_0': {
                                        'innerfold_0': {'train': indices_train, 'val': indices_val},
                                        ...
                                        'innerfold_n': {'train': indices_train, 'val': indices_val},
                                        'test': test_indices
                                        },
                                    ...
                                    'outerfold_m': {
                                        'innerfold_0': {'train': indices_train, 'val': indices_val},
                                        ...
                                        'innerfold_n': {'train': indices_train, 'val': indices_val},
                                        'test': test_indices
                                        }
                                    },
                            ...
                            }
                    'cv-test': {
                            '#folds-test_percentage': {
                                    'outerfold_0': {
                                        'innerfold_0': {'train': indices_train, 'val': indices_val},
                                        ...
                                        'innerfold_n': {'train': indices_train, 'val': indices_val},
                                        'test': test_indices
                                        }
                                    },
                            ...
                            }
                    'train-val-test': {
                            'train_percentage-val_percentage-test_percentage': {
                                    'outerfold_0': {
                                        'innerfold_0': {'train': indices_train, 'val': indices_val},
                                        'test': test_indices
                                        }
                                    },
                            ...
                            }
                    }

    Standard values for the data splits:

    - folds (inner-/outerfolds for 'nested-cv' and folds for 'cv-test'): 5
    - test percentage (for 'cv-test' and 'train-val-test'): 20
    - val percentage (for 'train-val-test'): 20

    :param data_dir: data directory where the dataset is stored
    :param dataset_name: name of the dataset .csv file to use
    :param datasplit: datasplit to use. Options are: nested-cv, cv-test, train-val-test
    :param n_outerfolds: number of outerfolds relevant for nested-cv
    :param n_innerfolds: number of folds relevant for nested-cv and cv-test
    :param test_set_size_percentage: size of the test set relevant for cv-test and train-val-test
    :param val_set_size_percentage: size of the validation set relevant for train-val-test
    """
    param_nested = [[5, 5]]  # standard values for outer and inner folds for nested-cv
    param_cv = [[5, 20]]  # standard number of folds and test percentage for cv-test
    param_tvt = [[20, 20]]  # standard train and val percentages for train-val-test split
    param_nested = check_datasplit_user_input(user_datasplit=datasplit,
                                              user_n_outerfolds=n_outerfolds, user_n_innerfolds=n_innerfolds,
                                              user_test_set_size_percentage=test_set_size_percentage,
                                              user_val_set_size_percentage=val_set_size_percentage,
                                              datasplit='nested-cv', param_to_check=param_nested)
    param_cv = check_datasplit_user_input(user_datasplit=datasplit,
                                          user_n_outerfolds=n_outerfolds, user_n_innerfolds=n_innerfolds,
                                          user_test_set_size_percentage=test_set_size_percentage,
                                          user_val_set_size_percentage=val_set_size_percentage,
                                          datasplit='cv-test', param_to_check=param_cv)
    param_tvt = check_datasplit_user_input(user_datasplit=datasplit,
                                           user_n_outerfolds=n_outerfolds, user_n_innerfolds=n_innerfolds,
                                           user_test_set_size_percentage=test_set_size_percentage,
                                           user_val_set_size_percentage=val_set_size_percentage,
                                           datasplit='train-val-test', param_to_check=param_tvt)

    dataset_raw = pd.read_csv(data_dir.joinpath(dataset_name))
    y = dataset_raw['label_binary'].to_numpy()
    with h5py.File(data_dir.joinpath('index_file_' + dataset_name.split('.')[0] + '.h5'), 'w') as f:
        # all data needed to redo matching of X and y and to create new mafs and new data splits
        data = f.create_group('full_data')
        data.create_dataset('all_sample_ids',
                            data=dataset_raw['meta_protein_id'].astype(bytes), chunks=True, compression="gzip")
        data.create_dataset('y', data=y, chunks=True, compression="gzip")

        # create and save standard data splits and splits according to user input
        dsplit = f.create_group('datasplits')
        nest = dsplit.create_group('nested-cv')
        for elem in param_nested:
            n = nest.create_group(helper_functions.get_subpath_for_datasplit(datasplit='nested-cv',
                                                                             datasplit_params=elem))
            index_dict = check_train_test_splits(y=y, datasplit='nested-cv', datasplit_params=elem)
            for outer in range(elem[0]):
                o = n.create_group(f'outerfold_{outer}')
                o.create_dataset('test', data=index_dict[f'outerfold_{outer}_test'], chunks=True, compression="gzip")
                for inner in range(elem[1]):
                    i = o.create_group(f'innerfold_{inner}')
                    i.create_dataset('train', data=index_dict[f'outerfold_{outer}'][f'fold_{inner}_train'], chunks=True,
                                     compression="gzip")
                    i.create_dataset('val', data=index_dict[f'outerfold_{outer}'][f'fold_{inner}_test'], chunks=True,
                                     compression="gzip")
        cv = dsplit.create_group('cv-test')
        for elem in param_cv:
            index_dict, test = check_train_test_splits(y=y, datasplit='cv-test', datasplit_params=elem)
            n = cv.create_group(helper_functions.get_subpath_for_datasplit(datasplit='cv-test', datasplit_params=elem))
            o = n.create_group('outerfold_0')
            o.create_dataset('test', data=test, chunks=True, compression="gzip")
            for fold in range(elem[0]):
                i = o.create_group(f'innerfold_{fold}')
                i.create_dataset('train', data=index_dict[f'fold_{fold}_train'], chunks=True, compression="gzip")
                i.create_dataset('val', data=index_dict[f'fold_{fold}_test'], chunks=True, compression="gzip")
        tvt = dsplit.create_group('train-val-test')
        for elem in param_tvt:
            train, val, test = check_train_test_splits(y=y, datasplit='train-val-test', datasplit_params=elem)
            n = tvt.create_group(helper_functions.get_subpath_for_datasplit(datasplit='train-val-test',
                                                                            datasplit_params=elem))
            o = n.create_group('outerfold_0')
            o.create_dataset('test', data=test, chunks=True, compression="gzip")
            i = o.create_group('innerfold_0')
            i.create_dataset('train', data=train, chunks=True, compression="gzip")
            i.create_dataset('val', data=val, chunks=True, compression="gzip")


def check_datasplit_user_input(user_datasplit: str, user_n_outerfolds: int, user_n_innerfolds: int,
                               user_test_set_size_percentage: int, user_val_set_size_percentage: int,
                               datasplit: str, param_to_check: list) -> list:
    """
    Check if user input of data split parameters differs from standard values.
    If it does, add input to list of parameters

    :param user_datasplit: datasplit specified by the user
    :param user_n_outerfolds: number of outerfolds relevant for nested-cv specified by the user
    :param user_n_innerfolds: number of folds relevant for nested-cv and cv-test specified by the user
    :param user_test_set_size_percentage: size of the test set relevant for cv-test and train-val-test specified by the user
    :param user_val_set_size_percentage: size of the validation set relevant for train-val-test specified by the user
    :param datasplit: type of data split
    :param param_to_check: standard parameters to compare to

    :return: adapted list of parameters
    """
    if datasplit == 'nested-cv':
        user_input = [user_n_outerfolds, user_n_innerfolds]
    elif datasplit == 'cv-test':
        user_input = [user_n_innerfolds, user_test_set_size_percentage]
    elif datasplit == 'train-val-test':
        user_input = [user_val_set_size_percentage, user_test_set_size_percentage]
    else:
        raise Exception('Only accept nested-cv, cv-test or train-val-test as data splits.')
    if user_datasplit == datasplit and user_input not in param_to_check:
        param_to_check.append(user_input)
    return param_to_check


def check_train_test_splits(y: np.array, datasplit: str, datasplit_params: list):
    """
    Create stratified train-test splits. Continuous values will be grouped into bins and stratified according to those.

    Datasplit parameters:

    - nested-cv: [n_outerfolds, n_innerfolds]
    - cv-test: [n_innerfolds, test_set_size_percentage]
    - train-val-test: [val_set_size_percentage, train_set_size_percentage]

    :param datasplit: type of datasplit ('nested-cv', 'cv-test', 'train-val-test')
    :param y: array with phenotypic values for stratification
    :param datasplit_params: parameters to use for split

    :return: dictionary respectively arrays with indices
    """
    y_binned = make_bins(y=y, datasplit=datasplit, datasplit_params=datasplit_params)
    if datasplit == 'nested-cv':
        return make_nested_cv(y=y_binned, outerfolds=datasplit_params[0], innerfolds=datasplit_params[1])
    elif datasplit == 'cv-test':
        x_train, x_test, y_train = make_train_test_split(y=y_binned, test_size=datasplit_params[1], val=False)
        cv_dict = make_stratified_cv(x=x_train, y=y_train, split_number=datasplit_params[0])
        return cv_dict, x_test
    elif datasplit == 'train-val-test':
        return make_train_test_split(y=y_binned, test_size=datasplit_params[1], val_size=datasplit_params[0], val=True)
    else:
        raise Exception('Only accept nested-cv, cv-test or train-val-test as data splits.')


def make_bins(y: np.array, datasplit: str, datasplit_params: list) -> np.array:
    """
    Create bins of continuous values for stratification.

    Datasplit parameters:

    - nested-cv: [n_outerfolds, n_innerfolds]
    - cv-test: [n_innerfolds, test_set_size_percentage]
    - train-val-test: [val_set_size_percentage, train_set_size_percentage]

    :param y: array containing phenotypic values
    :param datasplit: train test split to use
    :param datasplit_params: parameters to use for split

    :return: binned array
    """
    if helper_functions.test_likely_categorical(y):
        return y.astype(int)
    else:
        if datasplit == 'nested-cv':
            tmp = len(y)/(datasplit_params[0] + datasplit_params[1])
        elif datasplit == 'cv-test':
            tmp = len(y)*(1-datasplit_params[1]/100)/datasplit_params[0]
        else:
            tmp = len(y)/10 + 1

        number_of_bins = min(int(tmp) - 1, 10)
        edges = np.percentile(y, np.linspace(0, 100, number_of_bins)[1:])
        y_binned = np.digitize(y, edges, right=True)
        return y_binned


def make_nested_cv(y: np.array, outerfolds: int, innerfolds: int) -> dict:
    """
    Create index dictionary for stratified nested cross validation with the following structure:

    .. code-block:: python

        {
            'outerfold_0_test': test_indices,
            'outerfold_0': {
                'fold_0_train': innerfold_0_train_indices,
                'fold_0_test': innerfold_0_test_indices,
                ...
                'fold_n_train': innerfold_n_train_indices,
                'fold_n_test': innerfold_n_test_indices
            },
            ...
            'outerfold_m_test': test_indices,
            'outerfold_m': {
                'fold_0_train': innerfold_0_train_indices,
                'fold_0_test': innerfold_0_test_indices,
                ...
                'fold_n_train': innerfold_n_train_indices,
                'fold_n_test': innerfold_n_test_indices
            }
        }

    :param y: target values grouped in bins for stratification
    :param outerfolds: number of outer folds
    :param innerfolds: number of inner folds

    :return: index dictionary
    """
    outer_cv = StratifiedKFold(n_splits=outerfolds, shuffle=True)
    index_dict = {}
    outer_fold = 0
    for train_index, test_index in outer_cv.split(np.zeros(len(y)), y):
        np.random.shuffle(test_index)
        index_dict[f'outerfold_{outer_fold}_test'] = test_index
        index_dict[f'outerfold_{outer_fold}'] = make_stratified_cv(x=train_index, y=y[train_index],
                                                                   split_number=innerfolds)
        outer_fold += 1
    return index_dict


def make_stratified_cv(x: np.array, y: np.array, split_number: int) -> dict:
    """
    Create index dictionary for stratified cross-validation with following structure:

    .. code-block:: python

        {
            'fold_0_train': fold_0_train_indices,
            'fold_0_test': fold_0_test_indices,
            ...
            'fold_n_train': fold_n_train_indices,
            'fold_n_test': fold_n_test_indices
        }

    :param x: whole train indices
    :param y: target values binned in groups for stratification
    :param split_number: number of folds

    :return: dictionary containing train and validation indices for each fold
    """
    cv = StratifiedKFold(n_splits=split_number, shuffle=True)
    index_dict = {}
    fold = 0
    for train_index, test_index in cv.split(x, y):
        np.random.shuffle(train_index)
        np.random.shuffle(test_index)
        index_dict[f'fold_{fold}_train'] = x[train_index]
        index_dict[f'fold_{fold}_test'] = x[test_index]
        fold += 1
    return index_dict


def make_train_test_split(y: np.array, test_size: int, val_size=None, val=False, random=42) \
        -> (np.array, np.array, np.array):
    """
    Create index arrays for stratified train-test, respectively train-val-test splits.

    :param y: target values grouped in bins for stratification
    :param test_size: size of test set as percentage value
    :param val_size: size of validation set as percentage value
    :param val: if True, function returns validation set additionally to train and test set
    :param random: controls shuffling of data

    :return: either train, val and test index arrays or train and test index arrays and corresponding binned target values
    """
    x = np.arange(len(y))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size/100, stratify=y, random_state=random)
    if not val:
        return x_train, x_test, y_train
    else:
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size/100, stratify=y_train,
                                                          random_state=random)
        return x_train, x_val, x_test


def read_fasta(filepath: pathlib.Path) -> (np.array, np.array):
    """
    Read protein FASTA files (.fasta, .txt), i.e. files that contain pais of protein ID and amino aid sequence:

    >protein1|some identifier|another identifier|
    MGKRVVIALGGNALQQRGQKGSYEEMMDNVRKTARQIAEIIARGYEVVIT
    MSTESEIAVRIRGIYSTALTKLLMDRGFKIVQPSDVIAERFGIEKSYEDF
    DVDIYDKNHGVTIVGTKVEAVKKVFEEEFIDVFFRKLPYKLHGIYKGLVV
    KRDDRFVYVDIGNVIGTVLIEELPDAAEGDEVVVQVKKHNVLPHLSTLIT
    >protein2|some identifier|another identifier|
    GLEDVYIDQTNICYIDGKEGKLYYRGYSVEELAELSTFEEVVYLLEIIAE

    Will return sequences and corresponding IDs (= the complete first line without '>')

    :param filepath: full path of FASTA file

    :return: sequences and corresponding IDs in two separate numpy arrays
    """
    with open(filepath, "r") as f:
        sequences = []
        protein_ids = []
        groups = (x[1] for x in groupby(f, lambda line: line[0] == ">"))
        for header in groups:
            pid = header.__next__().replace('>', '').strip()
            seq = "".join(s.strip() for s in groups.__next__())  # join all sequence lines to one.
            protein_ids.append(pid)
            sequences.append(seq)
    return np.array(sequences), np.array(protein_ids)


def remove_ambiguous_aa(sequences: np.array, protein_ids: np.array) -> (np.array, np.array):
    """
    Need to remove sequences with ambiguous amino acids in order to calculate peptide descriptors.

    :param sequences: array containing protein sequences to check
    :param protein_ids: array containing corresponding IDs

    :return: clean sequences and corresponding IDs in two separate numpy arrays
    """
    ambiguous_aa = ['B', 'J', 'U', 'X', 'Z', 'O']
    index = []
    for i, seq in enumerate(sequences):
        if not any(z in seq for z in ambiguous_aa):
            index.append(i)
    return sequences[index], protein_ids[index]


def add_features(raw_data: pd.DataFrame, fasta_file_paths: list, features_to_add: list) -> pd.DataFrame:
    """
    Add features, namely peptide descriptors described in features_to_add, to dataset

    :param raw_data: Dataframe containing raw data
    :param fasta_file_paths: full paths of all fasta files that are present in raw_data
    :param features_to_add: list of all features to add

    :return: dataframe containing dataset with all features
    """
    print('--- Adding features ---')
    feat_eng_data = raw_data.copy()
    # iterate over all features to get
    for feature_name in features_to_add:
        print('   ' + feature_name)
        # call function for basic descriptors
        if feature_name == 'Basic':
            features = add_basic_descriptors(raw_data=raw_data)
        # call iFeature for other descriptors
        else:
            features = None
            # iterate over all ids
            for fasta_file_path in fasta_file_paths:
                ifeature_protein = iFeatureOmegaCLI.iProtein(file=fasta_file_path)
                full_ids = [el[0] + '|' + el[2] + '|' + el[3] for el in ifeature_protein.fasta_list]
                ifeature_protein.get_descriptor(feature_name)
                new_features = ifeature_protein.encodings
                new_features.index = full_ids
                if features is None:
                    features = new_features.copy()
                else:
                    features = pd.concat([features, new_features])
            if feature_name in ['CTDC', 'CTDT', 'CTDD']:
                features = features.filter(
                    regex='PRAM900101|solventaccess|secondarystruct|charge|polarizability|polarity|normwaalsvolume'
                )
        # merge features with dataset
        features = features.add_prefix('feat_')
        features['meta_protein_id'] = features.index
        features.reset_index(drop=True, inplace=True)
        features.drop_duplicates(inplace=True, subset='meta_protein_id')
        print('   - n_feats: ' + str(features.shape[1] - 1))
        feat_eng_data = \
            pd.merge(feat_eng_data, features, how='left', on='meta_protein_id')
    return feat_eng_data


def add_basic_descriptors(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic descriptors such as molecule weight

    hydrophobicity from: https://www.genome.jp/entry/aaindex:PRAM900101
    vdW_volume and charge from: https://doi.org/10.1111/j.1399-3011.1988.tb01261.x


    :param raw_data: Dataframe containing raw data

    :return: dataframe containing basic descriptors to add
    """
    features = pd.DataFrame()
    features.index = raw_data['meta_protein_id']
    peptide_information = pd.DataFrame(
        index=['weight', 'charge', 'polar', 'n_benzene', 'hydrophobicity', 'vdW_volume'],
        data=get_basic_aa_info()
    )

    for index, sample in raw_data.iterrows():
        sequence = sample['seq_peptide']
        for descriptor, row in peptide_information.iterrows():
            sequence_info = [row[amino_acid] for amino_acid in sequence]
            if descriptor == 'weight':
                # subtract peptide bond water particle for molecular weight
                features.at[sample['meta_protein_id'], 'Mw'] = np.sum(sequence_info) - (len(sequence) - 1) * 18
            elif descriptor in ['hydrophobicity', 'vdW_volume']:
                # divide by sequence length for hydrophobicity and vdw
                features.at[sample['meta_protein_id'], descriptor] = np.sum(sequence_info) / len(sequence)
            elif descriptor == 'n_benzene':
                features.at[sample['meta_protein_id'], descriptor] = np.sum(sequence_info)
            elif descriptor == 'charge':
                features.at[sample['meta_protein_id'], 'charge_of_all'] = np.sum(sequence_info)
                features.at[sample['meta_protein_id'], 'pos_charge'] = sequence_info.count(1)
                features.at[sample['meta_protein_id'], 'neg_charge'] = sequence_info.count(-1)
            elif descriptor == 'polar':
                features.at[sample['meta_protein_id'], 'polar'] = sequence_info.count(1)
                features.at[sample['meta_protein_id'], 'unpolar'] = sequence_info.count(-1)

    return features


def get_basic_aa_info() -> dict:
    """
    Basic AA info such as weight

    hydrophobicity from: https://www.genome.jp/entry/aaindex:PRAM900101
    vdW_volume and charge from: https://doi.org/10.1111/j.1399-3011.1988.tb01261.x


    :return: dict containing basic info
    """
    # 'weight', 'charge', 'polar', 'n_benzene', 'hydrophobicity', 'vdW_volume'
    return {'A': np.array([89.1, 0, -1, 0, -6.7, 1]),
            'R': np.array([174.2, 1, 1, 0, 51.5, 6.13]),
            'N': np.array([132.1, 0, 1, 0, 20.1, 2.95]),
            'D': np.array([133.1, -1, 1, 0, 38.5, 2.78]),
            'C': np.array([121.2, 0, 1, 0, -8.4, 2.43]),
            'Q': np.array([146.2, 0, 1, 0, 17.2, 3.95]),
            'E': np.array([147.1, -1, 1, 0, 34.3, 3.78]),
            'G': np.array([75.1, 0, -1, 0, -4.2, 0]),
            'H': np.array([155.2, 1, 1, 0, -12.6, 4.66]),
            'I': np.array([131.2, 0, -1, 0, -13.0, 4]),
            'L': np.array([131.2, 0, -1, 0, -11.7, 4]),
            'K': np.array([146.2, 1, 1, 0, 36.8, 4.77]),
            'M': np.array([149.2, 0, -1, 0, -14.2, 4.43]),
            'F': np.array([165.2, 0, -1, 1, -15.5, 5.89]),
            'P': np.array([115.1, 0, -1, 0, 0.8, 2.72]),
            'S': np.array([105.1, 0, 1, 0, -2.5, 1.6]),
            'T': np.array([119.1, 0, 1, 0, -5.0, 2.6]),
            'W': np.array([204.2, 0, -1, 1, -7.9, 8.08]),
            'Y': np.array([181.2, 0, 1, 1, 2.9, 6.47]),
            'V': np.array([117.2, 0, -1, 0, -10.9, 3])
        }


def get_standardized_basic_aa_info() -> dict:
    """
    Standardized basic aa info

    :return: dict containing standardized basic aa info
    """
    aa_info = get_basic_aa_info()
    items = []

    for ind in range(len(aa_info['A'])):
        items.append([values[ind] for key, values in aa_info.items()])

    for key, values in aa_info.items():
        for ind in range(len(values)):
            aa_info[key][ind] = (aa_info[key][ind] - np.mean(items[ind])) / np.std(items[ind])

    return aa_info
