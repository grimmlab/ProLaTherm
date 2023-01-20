import pathlib
import torch
import re
from transformers import T5Tokenizer, T5EncoderModel
from collections import OrderedDict

import utils


class ProLaTherm(torch.nn.Module):
    """
    Implementation of our Protein Language Model-based Thermophilicity Predictor ProLaTherm:
        - Encoding of amino acid sequence
        - Protein language model-based feature extractor
        - Average pooling along sequence length
        - Head classifier
    """
    def __init__(self, dropout_rate: float = 0.2):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        pretrained_model = "Rostlab/prot_t5_xl_uniref50"
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model, do_lower_case=False)
        self.feat_ext = T5EncoderModel.from_pretrained(pretrained_model).to(device=self.device)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.average_pooling = AveragePoolProtEmbeds()
        self.head = HeadClassifier().to(device=self.device)
        utils.set_all_seeds()

    def get_input_ids_attention_masks(self, input_seqs):
        """
        Use tokenizer to get input ids and attention masks
        :param input_seqs: Input sequences
        :return: input ids and attention mask
        """
        seq = [" ".join(seq) for seq in input_seqs]
        seq = [re.sub(r"[UZOB]", "X", sequence) for sequence in seq]
        ids = self.tokenizer.batch_encode_plus(seq, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.device)
        return input_ids, attention_mask

    def forward(self, input_seqs):
        """
        Forward path consisting of encoding, feature extractor and head classifer and returning logits
        :param input_seqs: input sequences
        :return: logits of predictions
        """
        self.feat_ext.eval()
        self.dropout.eval()
        self.head.eval()
        input_ids, attention_mask = self.get_input_ids_attention_masks(input_seqs)
        with torch.no_grad():
            out = self.feat_ext(input_ids=input_ids, attention_mask=attention_mask)
        embedding = out.last_hidden_state
        seq_lens = [(attention_mask[seq_num] == 1).sum() for seq_num in range(attention_mask.size()[0])]
        X_embed = torch.zeros((embedding.size()[0], max(seq_lens), 1024), device=self.device)
        for seq_num in range(embedding.size()[0]):
            seq_len = (attention_mask[seq_num] == 1).sum()
            X_embed[seq_num, :seq_len - 1, :] = embedding[seq_num][:seq_len - 1][:]
        x = X_embed
        x = self.dropout(x)
        x = self.average_pooling(x)
        out = self.head(x)
        return out

    def predict(self, input_seqs):
        """
        Get predictions and scores for input sequences
        :param input_seqs: input sequences
        :return: predictions and scores
        """
        logits = self.forward(input_seqs)
        _, predictions = torch.max(logits, 1)
        scores = torch.nn.functional.softmax(logits, dim=1)[:, 1]
        return predictions, scores


class AveragePoolProtEmbeds(torch.nn.Module):
    def forward(self, x):
        """
        Average pool along sequence length
        :param x: input of dim (batch_size, seq_length, latent_dim)
        :return: average pooled result along seq_length
        """
        return x.mean(dim=1)


class HeadClassifier(torch.nn.Module):
    """
    Implementation of ProLaTherm's head classifier
    """
    def __init__(self, dropout_rate: float = 0.2):
        super().__init__()
        self.model = self.define_model(dropout_rate=0.2)
        self.reorder_load_state_dict(pathlib.Path().absolute().joinpath('assets/head_state_dict'))

    @staticmethod
    def define_model(dropout_rate: float):
        """
        Model definition of the head classifier
        :param dropout_rate: rate for dropout layer
        :return: model as torch Sequential
        """
        model = [
            torch.nn.Linear(in_features=1024, out_features=512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_features=512),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(in_features=512, out_features=2)
        ]

        return torch.nn.Sequential(*model)

    def reorder_load_state_dict(self, path_to_state_dict: pathlib.Path):
        """
        Set parameters of created head classifier with saved state dict"
        :param path_to_state_dict: path to the saved state dict of the head classifier
        """""
        state_dict = torch.load(path_to_state_dict)
        new_dict = OrderedDict()
        for old_key, new_key in zip(state_dict.keys(), self.model.state_dict().keys()):
            new_dict[new_key] = state_dict[old_key]
        self.model.load_state_dict(new_dict)

    def forward(self, x):
        """
        Forward path based on average pooled input
        :param x: average pooled along seq length input of dim (batch_size, latent_dim)
        :return: logits of predictions
        """
        return self.model(x)
