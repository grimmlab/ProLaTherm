import argparse

from . import results_analysis

if __name__ == "__main__":
    """
    Run to gather some overview files on the optimization results for the specified results directory
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-rd", "--results_dir", type=str,
                        help="Provide the full path of the directory where your results are stored "
                             "(name of the dataset)")
    args = vars(parser.parse_args())
    results_directory_dataset_name = args['results_dir']

    results_analysis.summarize_results_per_datasplit(
        results_directory_dataset_name=results_directory_dataset_name
    )
