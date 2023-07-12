import pandas as pd
import pathlib
import numpy as np
import sklearn.metrics

from ..utils import helper_functions, check_functions


def summarize_results_per_datasplit(results_directory_dataset_name: str):
    """
    Summarize the results for each datasplit

    :param results_directory_dataset_name: results directory at the level of the name of the genotype matrix
    """
    results_directory_dataset_name = pathlib.Path(results_directory_dataset_name)

    # Check user inputs
    print("Checking user inputs")
    if not check_functions.check_exist_directories(list_of_dirs=[results_directory_dataset_name]):
        raise Exception("See output above. Problems with specified directories")
    if not results_directory_dataset_name.parts[-2] == 'results':
        raise Exception("Problems with specified directory: " + str(results_directory_dataset_name) +
                        "\n Make sure the results directory is at the level fo the genotype matrix name.")

    subdirs = [fullpath.parts[-1]
               for fullpath in helper_functions.get_all_subdirectories_non_recursive(results_directory_dataset_name)]
    datasplit_patterns = set(['_'.join(path.split('_')[:2]) for path in subdirs])

    for pattern in list(datasplit_patterns):
        writer = pd.ExcelWriter(
            results_directory_dataset_name.joinpath('Detailed_results_summary_new' + pattern + '.xlsx'),
            engine='xlsxwriter'
        )
        overview_df = None
        print('----- Datasplit pattern ' + pattern + ' -----')
        print('Got results for ' + str(len([model for path in results_directory_dataset_name.glob(pattern + '*')
                                            for model in path.parts[-1].split('_')[3].split('+')])) +
              ' models.')
        for path in results_directory_dataset_name.glob(pattern + '*'):
            models = path.parts[-1].split('_')[2].split('+')
            for current_model in models:
                print('### Results for ' + current_model + ' ###')
                try:
                    results_file = list(path.glob('Results*.csv'))[0]
                    results = pd.read_csv(results_file)
                    results = results.loc[:, [current_model in col for col in results.columns]]
                    if 'nested' in pattern:
                        val_dict = None
                        eval_dict_std = result_string_to_dictionary(
                            result_string=results[current_model + '___eval_metrics'].iloc[-1])
                        eval_dict = result_string_to_dictionary(
                            result_string=results[current_model + '___eval_metrics'].iloc[-2]
                        )
                        runtime_dict_std = result_string_to_dictionary(
                            result_string=results[current_model + '___runtime_metrics'].iloc[-1]
                        )
                        runtime_dict = result_string_to_dictionary(
                            result_string=results[current_model + '___runtime_metrics'].iloc[-2]
                        )
                        """
                        # insert missing test metrics
                        bacc = []
                        spec = []
                        for of_fold_path in list(path.glob('outer*')):
                            final_model_test_results = \
                                pd.read_csv(list(of_fold_path.joinpath(current_model).glob('*test_results*'))[0])
                            y_true_test_res = final_model_test_results['y_true_test'].dropna().tolist()
                            y_pred_test_res = final_model_test_results['y_pred_test'].dropna().tolist()
                            tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true_test_res, y_pred_test_res).ravel()
                            bacc.append(
                                sklearn.metrics.balanced_accuracy_score(y_true=y_true_test_res, y_pred=y_pred_test_res)
                            )
                            spec.append(tn / (tn + fp))
                        eval_dict_std['test_bacc'] = np.std(bacc)
                        eval_dict_std['test_specifity'] = np.std(spec)
                        eval_dict['test_bacc'] = np.mean(bacc)
                        eval_dict['test_specifity'] = np.mean(spec)
                        """
                        val_results_files = list(path.glob('*/' + current_model + '/validation_results*.csv'))
                        val_dict = {str(key).replace('test_', 'val_') + '_mean': [] for key, val in eval_dict.items()}
                        val_dict_list = []
                        for val_results_file in val_results_files:
                            val_results = pd.read_csv(val_results_file)
                            val_dict_onefold = {str(key).replace('test_', 'val_') + '_mean': [] for key, val in
                                                eval_dict.items()}
                            keys = [key[4:-5] for key in val_dict.keys()]
                            for col in list(val_results.columns):
                                for key in keys:
                                    if key in col:
                                        val_dict['val_' + key + '_mean'].append(float(val_results.at[0, col]))
                                        val_dict_onefold['val_' + key + '_mean'].append(float(val_results.at[0, col]))
                            # insert missing val results
                            """
                            for fold in range(5):
                                true = val_results['innerfold_' + str(fold) + '_val_true'].dropna().tolist()
                                pred = val_results['innerfold_' + str(fold) + '_val_pred'].dropna().tolist()
                                tn, fp, fn, tp = sklearn.metrics.confusion_matrix(true, pred).ravel()
                                val_dict_onefold['val_' + 'bacc' + '_mean'].append(
                                    float(sklearn.metrics.balanced_accuracy_score(y_true=true, y_pred=pred)))
                                val_dict_onefold['val_' + 'specifity' + '_mean'].append(float(tn / (tn + fp)))
                                val_dict['val_' + 'bacc' + '_mean'].append(
                                    float(sklearn.metrics.balanced_accuracy_score(y_true=true, y_pred=pred)))
                                val_dict['val_' + 'specifity' + '_mean'].append(float(tn / (tn + fp)))
                            """
                            val_dict_list.append(val_dict_onefold)
                    else:
                        eval_dict = result_string_to_dictionary(
                            result_string=results[current_model + '___eval_metrics'][0]
                        )
                        runtime_dict = result_string_to_dictionary(
                            result_string=results[current_model + '___runtime_metrics'][0]
                        )
                        # insert missing test results
                        """
                        final_model_test_results = \
                            pd.read_csv(list(path.joinpath(current_model).glob('*test_results*'))[0])
                        y_true_test_res = final_model_test_results['y_true_test'].dropna().tolist()
                        y_pred_test_res = final_model_test_results['y_pred_test'].dropna().tolist()
                        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true_test_res, y_pred_test_res).ravel()
                        eval_dict['test_bacc'] = \
                            sklearn.metrics.balanced_accuracy_score(y_true=y_true_test_res, y_pred=y_pred_test_res)
                        eval_dict['test_specifity'] = tn / (tn + fp)
                        """
                        val_results_file = list(path.joinpath(current_model).glob('validation_results*.csv'))[0]
                        val_results = pd.read_csv(val_results_file)
                        val_dict = {str(key).replace('test_', 'val_') + '_mean': [] for key, val in eval_dict.items()}
                        val_dict_list = None
                        keys = [key[4:-5] for key in val_dict.keys()]
                        for col in list(val_results.columns):
                            for key in keys:
                                if key in col:
                                    val_dict['val_' + key + '_mean'].append(float(val_results.at[0, col]))
                        """
                        for fold in range(5):
                            true = val_results['innerfold_' + str(fold) + '_val_true'].dropna().tolist()
                            pred = val_results['innerfold_' + str(fold) + '_val_pred'].dropna().tolist()
                            tn, fp, fn, tp = sklearn.metrics.confusion_matrix(true, pred).ravel()
                            val_dict['val_' + 'bacc' + '_mean'].append(
                                float(sklearn.metrics.balanced_accuracy_score(y_true=true, y_pred=pred)))
                            val_dict['val_' + 'specifity' + '_mean'].append(float(tn / (tn + fp)))
                        """
                    excel_save_str = current_model if current_model != 'transformerreducedposembed' \
                        else 'transredposembed'
                    results.to_excel(writer, sheet_name=excel_save_str + '_results', index=False)
                    eval_dict = {str(key) + '_mean': val for key, val in eval_dict.items()}
                    runtime_dict = {str(key) + '_mean': val for key, val in runtime_dict.items()}
                    new_row = {'model': current_model}
                    new_row.update(eval_dict)
                    new_row.update(runtime_dict)
                    if 'nested' in pattern:
                        eval_dict_std = {str(key) + '_std': val for key, val in eval_dict_std.items()}
                        runtime_dict_std = {str(key) + '_std': val for key, val in runtime_dict_std.items()}
                        new_row.update(eval_dict_std)
                        new_row.update(runtime_dict_std)
                    if val_dict is not None:
                        val_dict_std = {}
                        val_dict_mean = {}
                        for key, val in val_dict.items():
                            val_dict_mean[key] = np.mean(val_dict[key])
                            val_dict_std[key.replace('mean', 'std')] = np.std(val_dict[key])
                        new_row.update(val_dict_mean)
                        new_row.update(val_dict_std)
                    if val_dict_list is not None:
                        for fold, val_dict in enumerate(val_dict_list):
                            val_dict_std = {}
                            val_dict_mean = {}
                            for key, val in val_dict.items():
                                val_dict_mean['of' + str(fold) + '_' + key] = np.mean(val_dict[key])
                                val_dict_std['of' + str(fold) + '_' + key.replace('mean', 'std')] = \
                                    np.std(val_dict[key])
                            new_row.update(val_dict_mean)
                            new_row.update(val_dict_std)
                    if overview_df is None:
                        overview_df = pd.DataFrame(new_row, index=[0])
                    else:
                        overview_df = pd.concat([overview_df, pd.DataFrame(new_row, index=[0])],
                                                ignore_index=True)
                except Exception as exc:
                    print('No Results File')
                    continue
                if 'nested' in pattern:
                    for outerfold_path in path.glob('outerfold*'):
                        runtime_file = pd.read_csv(
                            outerfold_path.joinpath(current_model, current_model + '_runtime_overview.csv')
                        )
                        runtime_file.to_excel(
                            writer, sheet_name=excel_save_str + '_of' + outerfold_path.parts[-1].split('_')[-1]
                                               + '_runtime',
                            index=False
                        )
                else:
                    runtime_file = \
                        pd.read_csv(path.joinpath(current_model, current_model + '_runtime_overview.csv'))
                    runtime_file.to_excel(writer, sheet_name=excel_save_str + '_runtime', index=False)
        overview_df.to_excel(writer, sheet_name='Overview_results', index=False)
        overview_df.to_csv(results_directory_dataset_name.joinpath('Results_summary_' + pattern + '.csv'))
        writer.sheets['Overview_results'].activate()
        writer.save()


def result_string_to_dictionary(result_string: str) -> dict:
    """
    Convert result string saved in a .csv file to a dictionary

    :param result_string: string from .csv file

    :return: dictionary with info from .csv file
    """
    if 'roc_list' in result_string:
        result_string = \
            result_string[:result_string.find('test_roc_list_fpr')] + result_string[result_string.find('test_roc_auc'):]
    elif 'prc_list' in result_string:
        result_string = \
            result_string[:result_string.find('test_prc_list_prec')] + result_string[result_string.find('test_prc_auc'):]
    key_value_strings = result_string.split('\\')[0][2:-2].replace('\'', '').split(',')
    dict_result = {}
    for key_value_string in key_value_strings:
        key = key_value_string.split(':')[0].strip()
        value = key_value_string.split(':')[1].strip()
        try:
            value = float(value)
            value = int(value) if value == int(value) else value
        except:
            value = value
        dict_result[key] = value if value != 'True' else True
    return dict_result

