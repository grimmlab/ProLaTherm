import argparse
import pandas as pd
import pathlib
from transformers import T5Tokenizer, T5EncoderModel
import torch
import h5py

if __name__ == '__main__':

    # User Input #
    parser = argparse.ArgumentParser()
    # Input Params #
    parser.add_argument("-dd", "--data_dir", type=str,
                        default='/bit_storage/Projects/ProteinThermPred/final_datasets/new_data/', #'/myhome/data/', #'/bit_storage/Projects/ProteinThermPred/final_datasets/',
                        help="Provide the full path of your data directory (that contains the geno- and phenotype "
                             "files).")
    parser.add_argument("-sd", "--save_dir", type=str, default='/myhome/work/',
                        help="Provide the full path of the directory in which you want to save your results. "
                             "Default is same as data_dir")
    parser.add_argument("-ds", "--dataset_name", type=str, default='ProtThermPred_fulldataset_evidence_only_clean.csv',
                        help="specify the name of the dataset to be used. Has to be a .csv file in our unified format"
                             "If working for the first time, you can also specify the options --fasta_thermo and "
                             "--fasta_nonthermo instead to generate the unified .csv format using fasta files."
                             "Needs to be located in the specified data_dir."
                        )
    parser.add_argument("-mod", "--pretrained_model", type=str, default="Rostlab/prot_t5_xl_uniref50")
    args = vars(parser.parse_args())

    pretrained_model = args["pretrained_model"]
    save_dir = pathlib.Path(args["save_dir"])
    data_dir = pathlib.Path(args["data_dir"])
    dataset_name = args["dataset_name"]

    tokenizer = T5Tokenizer.from_pretrained(pretrained_model, do_lower_case=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = T5EncoderModel.from_pretrained(pretrained_model)
    model = model.to(device)
    model = model.eval()
    dataset_raw = pd.read_csv(data_dir.joinpath(dataset_name))
    num_samples = dataset_raw.shape[0]
    prot_ids_full = list(dataset_raw['meta_protein_id'])
    seq_full = [" ".join(seq) for seq in dataset_raw["seq_peptide"]]
    start = 0
    step_size = 10
    for i in range(step_size, num_samples, step_size):
        end = i if i + step_size < num_samples else num_samples
        print(start)
        print(end)
        prot_ids = prot_ids_full[start:end]
        seq = seq_full[start:end]

        ids = tokenizer.batch_encode_plus(seq, add_special_tokens=True, padding=True)

        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        with torch.no_grad():
             out = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = out.last_hidden_state.cpu().numpy()
        with h5py.File(
                data_dir.joinpath('embeddings_' + dataset_name.split('.')[0] + '.h5'), 'a') as f:
            for seq_num in range(len(embedding)):
                seq_len = (attention_mask[seq_num] == 1).sum()
                seq_emd = embedding[seq_num][:seq_len - 1]
                protein_id = prot_ids[seq_num]
                print(protein_id)
                print(seq_emd.shape)
                f.create_dataset('/' + pretrained_model.split('/')[1] + '/' + protein_id + '/embeddings', data=seq_emd,
                                 chunks=True, compression="gzip")
        start = end
