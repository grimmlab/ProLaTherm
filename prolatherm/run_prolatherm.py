import pandas as pd
import pathlib
import argparse
import utils
import warnings

import prolatherm


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    base_path = pathlib.Path().absolute()
    # Input Params #
    parser.add_argument("-df", "--data_dir_fasta", type=str,
                        default=base_path.joinpath('assets', 'dummy_data.fasta'),
                        help="Provide the full path to the fasta file you want to get predictions for."
                             "Default is the dummy_data.fasta we provide in prolatherm/assets")
    parser.add_argument("-sd", "--save_dir", type=str, default=base_path.parent,
                        help="Provide the full path of the directory in which you want to save your results. "
                             "Default is in the root directory of your repository.")
    parser.add_argument("-ng", "--no_gpu", type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        default=False, help="Do not use GPU if set True. Default: False")
    args = vars(parser.parse_args())
    data_dir_fasta = pathlib.Path(args["data_dir_fasta"])
    save_dir = pathlib.Path(args["save_dir"])

    # Check input arguments
    print("Checking input arguments")
    if not data_dir_fasta.is_file():
        raise Exception("Specified fasta file does not exist: " + str(data_dir_fasta))
    if not save_dir.exists():
        raise Exception("Specified save directory does not exist: " + str(save_dir))
    print("Input arguments valid")

    # Read fasta file
    print("Reading fasta file " + str(data_dir_fasta))
    sequences, protein_ids = utils.read_fasta(data_dir_fasta)
    if len(sequences) == 0:
        raise Exception("Fasta file is empty or does not match format")

    # Load prediction model
    print("Load ProLaTherm model")
    pred_model = prolatherm.ProLaTherm(no_gpu=args["no_gpu"])

    # Run prediction pipeline in batches of size 10
    start = 0
    step_size = 10
    num_samples = len(sequences)
    preds = []
    scores = []
    print("Start prediction pipeline")
    for i in range(0, num_samples, step_size):
        end = i + step_size if i + step_size < num_samples else num_samples
        seqs = sequences[start:end]
        if start+1 == end:
            seqs = [seqs]
        print("Running prediction for sequences " + str(start+1) + ' to ' + str(end) + " of in total " + str(num_samples))
        pred, score = pred_model.predict(seqs)
        preds.extend(pred.flatten().tolist())
        scores.extend(score.flatten().tolist())
        start = end
    print("Finished. Creating results file.")

    # Save results
    results = pd.DataFrame(columns=["IDs", "aa-seq", "prediction_binary", "prediction", "score"])
    results["IDs"] = protein_ids
    results["prediction_binary"] = preds
    results["score"] = scores
    results["prediction"] = \
        results["prediction_binary"].apply(lambda x: "thermophilic" if x == 1 else "non-thermophilic")
    results["aa-seq"] = sequences
    results_file_path = save_dir.joinpath("ProLaTherm_Predictions_" + str(data_dir_fasta.stem) + ".csv")
    results.to_csv(results_file_path, index=False)
    print("Saved results at: " + str(results_file_path))
    print("Here are the final predictions:")
    print(results)
