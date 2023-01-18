import h5py
import numpy as np
import sklearn.preprocessing
import pathlib
import pandas as pd
import math

from ..utils import helper_functions
from . import raw_data_functions, encoding_functions


class Dataset:
    """
    Class containing dataset ready for optimization.

    **Attributes**

        - featureset (*str*): the featureset to use
        - X_full (*numpy.array*): all features
        - y_full (*numpy.array*): all target values
        - sample_ids_full (*numpy.array*): all sample ids
        - feature_names (*numpy.array*): all feature names
        - datasplit (*str*): datasplit to use
        - datasplit_indices (*dict*): dictionary containing all indices for the specified datasplit

    :param data_dir: data directory where the dataset is stored
    :param dataset_name: name of the dataset to use
    :param datasplit: datasplit to use. Options are: nested-cv, cv-test, train-val-test
    :param n_outerfolds: number of outerfolds relevant for nested-cv
    :param n_innerfolds: number of folds relevant for nested-cv and cv-test
    :param test_set_size_percentage: size of the test set relevant for cv-test and train-val-test
    :param val_set_size_percentage: size of the validation set relevant for train-val-test
    :param featureset: the featureset to use
    """

    def __init__(self, data_dir: pathlib.Path, dataset_name: str, datasplit: str, n_outerfolds: int, n_innerfolds: int,
                 test_set_size_percentage: int, val_set_size_percentage: int, featureset: str, pad_to_32: bool = False):
        self.pad_to_32 = pad_to_32
        self.featureset = featureset
        self.datasplit = datasplit
        # self.embedding_file = data_dir.joinpath('embeddings_' + dataset_name.split('.')[0] + '.h5')
        self.index_file_name = self.get_index_file_name(dataset_name=dataset_name)
        self.X_full, self.y_full, self.sample_ids_full, self.feature_names, self.seq_lenghts = self.load_data(
            data_dir=data_dir, dataset_name=dataset_name
        )
        self.datasplit_indices = self.load_datasplit_indices(
            data_dir=data_dir, n_outerfolds=n_outerfolds, n_innerfolds=n_innerfolds,
            test_set_size_percentage=test_set_size_percentage, val_set_size_percentage=val_set_size_percentage
        )
        self.check_datasplit(
            n_outerfolds=1 if datasplit != 'nested-cv' else n_outerfolds,
            n_innerfolds=1 if datasplit == 'train-val-test' else n_innerfolds
        )

    def load_data(self, data_dir: pathlib.Path, dataset_name: str, pretrained: str = 'prot_t5_xl_uniref50') \
            -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Load the full dataset, based on specified self.features.

        :param data_dir: data directory where the dataset is stored
        :param dataset_name: name of the csv file to use
        :param pretrained: name of pretrained model for embeddings

        :return: X matrix, y vector, sample_ids, feature_names
        """
        print('Load raw data')
        with h5py.File(data_dir.joinpath(self.index_file_name), "r") as f:
            # Load information from index file
            y = f['full_data/y'][:]
            if helper_functions.test_likely_categorical(y):
                if y.dtype.type is np.float64:
                    y = y.astype(int)
                y = sklearn.preprocessing.LabelEncoder().fit_transform(y)
            sample_ids = f['full_data/all_sample_ids'][:].astype(str)

        dataset_raw = pd.read_csv(data_dir.joinpath(dataset_name))
        if self.featureset == 'features':
            filter = 'feat_'
            X = dataset_raw.filter(regex=filter + '*')
            feature_names = np.array(X.columns)
            X = X.to_numpy()
            seq_lenghts = None
        elif self.featureset in ['sequence', 'aa_desc_matrix']:
            X = dataset_raw['seq_peptide']
            X = X.to_numpy()
            seq_lenghts = np.array([len(x) for x in X])
            max_len = int(max([len(element) for element in X]))
            if self.featureset == 'sequence':
                seq_len_to_pad = math.ceil(max_len / 32) * 32 if self.pad_to_32 else max_len
                X_enc = np.zeros((len(sample_ids), seq_len_to_pad))
                for index in range(len(X)):
                    X_enc[index] = \
                        encoding_functions.get_indexed_vector_from_sequence(sequence=X[index], max_len=seq_len_to_pad)
            else:
                X_enc = np.zeros((len(sample_ids), max_len, len(raw_data_functions.get_basic_aa_info()['A'])))
                for index in range(len(X)):
                    X_enc[index, :, :] = \
                        encoding_functions.get_aa_info_encoded_sequence(sequence=X[index], max_len=max_len)
            X = X_enc
            feature_names = np.array(self.featureset)
        elif self.featureset == 'pretrained':
            seq_lenghts = np.array([len(x) for x in dataset_raw['seq_peptide'].to_numpy()])
            # X = np.array(dataset_raw.index)  # will be loaded on the fly due to huge size
            X = helper_functions.load_embeddings_for_prot_ids(
                prot_ids=list(sample_ids), #[:50],
                embedding_file=data_dir.joinpath('embeddings_' + dataset_name.split('.')[0] + '.h5'),
                max_len=int(max(seq_lenghts))
            )
            feature_names = np.array(self.featureset)
        else:
            filter = 'struc_' #TODO: anpassen sobald 3d vorhanden

        return X, np.reshape(y, (-1, 1)), np.reshape(sample_ids, (-1, 1)), feature_names, seq_lenghts

    def load_datasplit_indices(self, data_dir: pathlib.Path, n_outerfolds: int, n_innerfolds: int,
                               test_set_size_percentage: int, val_set_size_percentage: int) -> dict:
        """
        Load the datasplit indices saved during file unification.

        Structure:

        .. code-block:: python

            {
                'outerfold_0': {
                    'innerfold_0': {'train': indices_train, 'val': indices_val},
                    'innerfold_1': {'train': indices_train, 'val': indices_val},
                    ...
                    'innerfold_n': {'train': indices_train, 'val': indices_val},
                    'test': test_indices
                    },
                ...
                'outerfold_m': {
                    'innerfold_0': {'train': indices_train, 'val': indices_val},
                    'innerfold_1': {'train': indices_train, 'val': indices_val},
                    ...
                    'innerfold_n': {'train': indices_train, 'val': indices_val},
                    'test': test_indices
                    }
            }

        Caution: The actual structure depends on the datasplit specified by the user,
        e.g. for a train-val-test split only 'outerfold_0' and its subelements 'innerfold_0' and 'test' exist.

        :param data_dir: data directory where the phenotype and genotype matrix are stored
        :param n_outerfolds: number of outerfolds relevant for nested-cv
        :param n_innerfolds: number of folds relevant for nested-cv and cv-test
        :param test_set_size_percentage: size of the test set relevant for cv-test and train-val-test
        :param val_set_size_percentage: size of the validation set relevant for train-val-test

        :return: dictionary with the above-described structure containing all indices for the specified data split
        """
        print('Load datasplit file')
        # construct variables for further process
        if self.datasplit == 'train-val-test':
            n_outerfolds = 1
            n_innerfolds = 1
            datasplit_params = [val_set_size_percentage, test_set_size_percentage]
        elif self.datasplit == 'cv-test':
            n_outerfolds = 1
            n_innerfolds = n_innerfolds
            datasplit_params = [n_innerfolds, test_set_size_percentage]
        elif self.datasplit == 'nested-cv':
            n_outerfolds = n_outerfolds
            n_innerfolds = n_innerfolds
            datasplit_params = [n_outerfolds, n_innerfolds]
        split_param_string = helper_functions.get_subpath_for_datasplit(datasplit=self.datasplit,
                                                                        datasplit_params=datasplit_params)

        datasplit_indices = {}
        with h5py.File(data_dir.joinpath(self.index_file_name), "r") as f:
            # load datasplit indices from index file to ensure comparability between different models
            for m in range(n_outerfolds):
                outerfold_path = \
                    f'datasplits/{self.datasplit}/{split_param_string}/outerfold_{m}/'
                datasplit_indices['outerfold_' + str(m)] = {'test': f[f'{outerfold_path}test/'][:]}
                for n in range(n_innerfolds):
                    datasplit_indices['outerfold_' + str(m)]['innerfold_' + str(n)] = \
                        {
                            'train': f[f'{outerfold_path}innerfold_{n}/train'][:],
                            'val': f[f'{outerfold_path}innerfold_{n}/val'][:]
                        }
        return datasplit_indices

    def check_datasplit(self, n_outerfolds: int, n_innerfolds: int):
        """
        Check if the datasplit is valid. Raise Exceptions if train, val or test sets contain same samples.

        :param n_outerfolds: number of outerfolds in datasplit_indices dictionary
        :param n_innerfolds: number of folds in datasplit_indices dictionary
        """
        all_sample_ids_test = []
        for j in range(n_outerfolds):
            sample_ids_test = set(self.sample_ids_full[self.datasplit_indices[f'outerfold_{j}']['test']].flatten())
            all_sample_ids_test.extend(sample_ids_test)
            for i in range(n_innerfolds):
                sample_ids_train = set(
                    self.sample_ids_full[self.datasplit_indices[f'outerfold_{j}'][f'innerfold_{i}']['train']].flatten()
                )
                sample_ids_val = set(
                    self.sample_ids_full[self.datasplit_indices[f'outerfold_{j}'][f'innerfold_{i}']['val']].flatten())
                if len(sample_ids_train.intersection(sample_ids_val)) != 0:
                    raise Exception(
                        'Something with the datasplit went wrong - the intersection of train and val samples is not '
                        'empty. Please check again.'
                    )
                if len(sample_ids_train.intersection(sample_ids_test)) != 0:
                    raise Exception(
                        'Something with the datasplit went wrong - the intersection of train and test samples is not '
                        'empty. Please check again.'
                    )
                if len(sample_ids_val.intersection(sample_ids_test)) != 0:
                    raise Exception(
                        'Something with the datasplit went wrong - the intersection of val and test samples is not '
                        'empty. Please check again.'
                    )
        if self.datasplit == 'nested-cv':
            if len(set(all_sample_ids_test).intersection(set(self.sample_ids_full.flatten()))) \
                    != len(set(self.sample_ids_full.flatten())):
                raise Exception('Something with the datasplit went wrong - '
                                'not all sample ids are in one of the outerfold test sets')
        print('Checked datasplit for all folds.')

    @staticmethod
    def get_index_file_name(dataset_name: str) -> str:
        """
        Get the name of the file containing the indices for maf filters and data splits

        :param dataset_name: name of the dataset to use

        :return: name of index file
        """
        return 'index_file_' + dataset_name.split('.')[0] + '.h5'
