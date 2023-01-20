import argparse
import pathlib

from thermpred.utils import helper_functions
from . import optim_pipeline

if __name__ == '__main__':
    """
    Run file to start the whole procedure:
            Parameter Plausibility Check
            Check and prepare data files
            Bayesian optimization for each chosen model
    """
    # User Input #
    parser = argparse.ArgumentParser()
    base_path = pathlib.Path().absolute()
    # Input Params #
    parser.add_argument("-dd", "--data_dir", type=str,
                        default=base_path.parent.joinpath('data/datasets_w_datasplits/'),
                        help="Provide the full path of your data directory (that contains the geno- and phenotype "
                             "files).")
    parser.add_argument("-sd", "--save_dir", type=str, default=base_path.parent,
                        help="Provide the full path of the directory in which you want to save your results. "
                             "Default is same as data_dir")
    parser.add_argument("-ds", "--dataset_name", type=str, default='ProtThermPred_fulldataset.csv',
                        help="specify the name of the dataset to be used. Has to be a .csv file in our unified format"
                             "If working for the first time, you can also specify the options --fasta_thermo and "
                             "--fasta_nonthermo instead to generate the unified .csv format using fasta files."
                             "Needs to be located in the specified data_dir."
                        )
    parser.add_argument("-fth", "--fasta_file_thermo", type=str, default=None,
                        help="specify the name of the fasta file containing thermophilic protein sequences"
                             "Needs to be located in the specified data_dir.")
    parser.add_argument("-fnth", "--fasta_file_nonthermo", type=str, default=None,
                        help="specify the name of the fasta file containing non-thermophilic protein sequences"
                             "Needs to be located in the specified data_dir.")
    parser.add_argument("-nnd", "--name_new_dataset", type=str, default=None,
                        help="specify the name of the new dataset (.csv file) derived from the two provided fasta files"
                             "Will be saved in the specified data_dir.")

    # Preprocess Params #
    parser.add_argument("-split", "--datasplit", type=str, default='nested-cv',
                        help="specify the data split to use: 'nested-cv' | 'cv-test' | 'train-val-test'"
                             "Default values are 5 folds, train-test-split to 80/20 and train-val-test to 60/20/20")
    parser.add_argument("-testperc", "--test_set_size_percentage", type=int, default=20,
                        help="specify the size of the test set in percentage. "
                             "Standard is 20, only relevant for 'cv-test' and 'train-val-test'")
    parser.add_argument("-valperc", "--val_set_size_percentage", type=int, default=20,
                        help="specify the size of the validation set in percentage. "
                             "Standard is 20, only relevant for 'train-val-test'")
    parser.add_argument("-of", "--n_outerfolds", type=int, default=3,
                        help="specify the number of outerfolds to use for 'nested_cv'"
                             "Standard is 5, only relevant for 'nested_cv'")
    parser.add_argument("-folds", "--n_innerfolds", type=int, default=5,
                        help="specify the number of innerfolds/folds to use for 'nested_cv' respectively 'cv-test'"
                             "Standard is 5, only relevant for 'nested_cv' and 'cv-test'")

    # Model and Optimization Params #
    parser.add_argument("-mod", "--models", nargs='+', type=str, default=['svm'],
                        help="specify the models to optimize: 'all' or naming according to source file name. "
                             "Multiple models can be selected by just naming multiple model names, "
                             "e.g. --models mlp xgboost. "
                             "The following are available: " + str(helper_functions.get_list_of_implemented_models()))
    parser.add_argument("-tr", "--n_trials", type=int, default=10,
                        help="number of trials for optuna")
    parser.add_argument("-sf", "--save_final_model", type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        default=True,
                        help="save the final model to hard drive "
                             "(caution: some models may use a lot of disk space, "
                             "unfitted models that can be retrained are already saved by default)")

    # Only relevant for Neural Networks #
    parser.add_argument("-bs", "--batch_size", type=int, default=None,
                        help="Only relevant for neural networks: define the batch size.")
    parser.add_argument("-ep", "--n_epochs", type=int, default=None,
                        help="Only relevant for neural networks: define the number of epochs. If nothing is specified,"
                             "it will be considered as a hyperparameter for optimization")

    args = vars(parser.parse_args())

    try:
        optim_pipeline.run(**args)
    except Exception as exc:
        print("Failure when running pipeline for ")
        print(args)
        print(exc)
