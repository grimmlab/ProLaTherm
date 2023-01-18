import torch

from . import _torch_model


class Lstm(_torch_model.TorchModel):
    """
    Implementation of a class for a LSTM

    See :obj:`~thermpred.model._base_model.BaseModel` and :obj:`~thermpred.model._torch_model.TorchModel` for more information on the attributes.
    """
    featureset = 'sequence'

    def define_model(self) -> torch.nn.Sequential:
        """
        Definition of an LSTM network.
        """
        model = []
        act_function = self.get_torch_object_for_string(string_to_get=self.suggest_hyperparam_to_optuna('act_function'))
        embedding_dim = 1024
        model.append(
            _torch_model.EmbedAndPackBlock(num_embeddings=self.size_alphabet+1, embedding_dim=embedding_dim)
        )
        p = self.suggest_hyperparam_to_optuna('dropout')
        input_size = embedding_dim
        hidden_size = 2**self.suggest_hyperparam_to_optuna('hidden_size_exp')
        model.append(
            _torch_model.LstmWithFlattenParams(
                input_size=input_size, hidden_size=hidden_size,
                num_layers=self.suggest_hyperparam_to_optuna('n_lstm_layers'),
                dropout=p, bidirectional=False)
        )
        model.append(_torch_model.ExtractTensor(bidirectional=False))
        model.append(torch.nn.Dropout(p=p))
        out_features = int(max(1, hidden_size * self.suggest_hyperparam_to_optuna('initial_units_factor')))
        model.append(torch.nn.Linear(in_features=hidden_size, out_features=out_features))
        model.append(act_function)
        model.append(torch.nn.BatchNorm1d(num_features=out_features))
        model.append(torch.nn.Dropout(p=p))
        model.append(torch.nn.Linear(in_features=out_features, out_features=self.n_outputs))
        return torch.nn.Sequential(*model)

    def define_hyperparams_to_tune(self) -> dict:
        """
        See :obj:`~thermpred.model._base_model.BaseModel` for more information on the format.

        See :obj:`~thermpred.model._torch_model.TorchModel` for more information on hyperparameters common for all torch models.
        """

        return {
            'n_lstm_layers': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 3
            },
            'hidden_size_exp': {
                'datatype': 'int',
                'lower_bound': 3,
                'upper_bound': 8
            },
            'initial_units_factor': {
                'datatype': 'float',
                'lower_bound': 0.5,
                'upper_bound': 1,
                'step': 0.05
            }
        }

