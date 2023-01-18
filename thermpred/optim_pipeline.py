import datetime
import pprint
import pathlib
import warnings
from optuna.exceptions import ExperimentalWarning

from thermpred.utils import check_functions, print_functions, helper_functions
from thermpred.preprocess import raw_data_functions, base_dataset
from thermpred.optimization import optuna_optim


def run(data_dir: str, dataset_name: str, fasta_file_thermo: str = None, fasta_file_nonthermo: str = None,
        name_new_dataset: str = None, save_dir: str = None,
        datasplit: str = 'nested-cv', n_outerfolds: int = 5, n_innerfolds: int = 5,
        test_set_size_percentage: int = 20, val_set_size_percentage: int = 20,
        models: list = None, n_trials: int = 100, save_final_model: bool = False,
        batch_size: int = 32, n_epochs: int = 100000, outerfold_number_to_run: int = None):
    """
    Run the whole optimization pipeline

    :param data_dir: data directory where the phenotype and genotype matrix are stored
    :param dataset_name: name of the dataset to use
    :param fasta_file_thermo: name of the fasta file containing the thermophilic proteins
    :param fasta_file_nonthermo: name of the fasta file containing the non-thermophilic proteins
    :param name_new_dataset: name of the new dataset based on the two provided fasta files
    :param save_dir: directory for saving the results. Default is None, so same directory as data_dir
    :param datasplit: datasplit to use. Options are: nested-cv, cv-test, train-val-test
    :param n_outerfolds: number of outerfolds relevant for nested-cv
    :param n_innerfolds: number of folds relevant for nested-cv and cv-test
    :param test_set_size_percentage: size of the test set relevant for cv-test and train-val-test
    :param val_set_size_percentage: size of the validation set relevant for train-val-test
    :param models: list of models that should be optimized
    :param n_trials: number of trials for optuna
    :param save_final_model: specify if the final model should be saved
    :param batch_size: batch size for neural network models
    :param n_epochs: number of epochs for neural network models
    :param outerfold_number_to_run: outerfold to run in case you do not want to run all
    """
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=ExperimentalWarning)

    if models is None:
        models = ['xgboost']
    # create Path
    data_dir = pathlib.Path(data_dir)
    # set save directory
    save_dir = data_dir if save_dir is None else pathlib.Path(save_dir)
    save_dir = save_dir if save_dir.is_absolute() else save_dir.resolve()
    if type(models) == list and models[0] == 'all':
        models = 'all'
    if type(models) != list and models != 'all':
        models = [models]

    # Checks and Raw Data Input Preparation #
    # Check all arguments
    check_functions.check_all_specified_arguments(arguments=locals())
    # prepare all data files
    raw_data_functions.prepare_data_files(
        data_dir=data_dir, dataset_name=dataset_name, fasta_file_non_thermo=fasta_file_nonthermo,
        fasta_file_thermo=fasta_file_thermo, name_new_dataset=name_new_dataset, datasplit=datasplit,
        n_outerfolds=n_outerfolds, n_innerfolds=n_innerfolds, test_set_size_percentage=test_set_size_percentage,
        val_set_size_percentage=val_set_size_percentage
    )
    if dataset_name is None:
        dataset_name = name_new_dataset + '.csv' if '.csv' not in name_new_dataset else name_new_dataset

    # Optimization Pipeline #
    helper_functions.set_all_seeds()
    models_to_optimize = helper_functions.get_list_of_implemented_models() if models == 'all' else models
    if len(models_to_optimize) > 1:
        models_to_optimize = helper_functions.sort_models_by_featureset(models_list=models_to_optimize)
    model_overview = {}
    only_ofn_postfix = '' if outerfold_number_to_run is None else 'Outerfold' + str(outerfold_number_to_run)
    models_start_time = \
        '+'.join(models_to_optimize) + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + only_ofn_postfix
    for optim_run, current_model_name in enumerate(models_to_optimize):
        featureset = helper_functions.get_mapping_name_to_class()[current_model_name].featureset
        if optim_run == 0:
            print('----- Starting dataset preparation -----')
            dataset = base_dataset.Dataset(
                data_dir=data_dir, dataset_name=dataset_name, datasplit=datasplit, n_outerfolds=n_outerfolds,
                n_innerfolds=n_innerfolds, test_set_size_percentage=test_set_size_percentage,
                val_set_size_percentage=val_set_size_percentage, featureset=featureset,
                pad_to_32=True if 'bigbird' in models_to_optimize else False
            )
            task = 'classification' if helper_functions.test_likely_categorical(dataset.y_full) else 'regression'
            print_functions.print_config_info(arguments=locals(), dataset=dataset, task=task)
        else:
            if dataset.featureset != featureset:
                print('----- Load new dataset  -----')
                dataset = base_dataset.Dataset(
                    data_dir=data_dir, dataset_name=dataset_name, datasplit=datasplit, n_outerfolds=n_outerfolds,
                    n_innerfolds=n_innerfolds, test_set_size_percentage=test_set_size_percentage,
                    val_set_size_percentage=val_set_size_percentage, featureset=featureset,
                    pad_to_32=True if 'bigbird' in models_to_optimize else False
                )
        optim_run = optuna_optim.OptunaOptim(
            save_dir=save_dir, dataset_name=dataset_name, n_outerfolds=n_outerfolds, n_innerfolds=n_innerfolds,
            val_set_size_percentage=val_set_size_percentage, test_set_size_percentage=test_set_size_percentage,
            n_trials=n_trials, save_final_model=save_final_model,
            batch_size=batch_size, n_epochs=n_epochs, task=task, models_start_time=models_start_time,
            current_model_name=current_model_name, dataset=dataset, outerfold_number_to_run=outerfold_number_to_run)
        print('### Starting Optuna Optimization for ' + current_model_name + ' ###')
        overall_results = optim_run.run_optuna_optimization()
        print('### Finished Optuna Optimization for ' + current_model_name + ' ###')
        model_overview[current_model_name] = overall_results

    print('# Optimization runs done for models ' + str(models_to_optimize))
    print('Results overview on the test set(s)')
    pprint.PrettyPrinter(depth=4).pprint(model_overview)
    path_overview_file = optim_run.base_path.parent.joinpath(
        'Results_overview_' + '_'.join(models) + only_ofn_postfix + '.csv')
    helper_functions.save_model_overview_dict(model_overview=model_overview, save_path=path_overview_file)
