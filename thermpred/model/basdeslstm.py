import torch

from . import _torch_model
from ..preprocess import raw_data_functions


class BasDesLstm(_torch_model.TorchModel):
    """
    Implementation of basic descriptor embedding and LSTM afterwards

    See :obj:`~thermpred.model._base_model.BaseModel` and :obj:`~thermpred.model._torch_model.TorchModel` for more information on the attributes.
    """
    featureset = 'aa_desc_matrix'

    def define_model(self) -> torch.nn.Sequential:
        """
        Definition of an CNN-LSTM network.

        Architecture:

            - tbd.

        """
        model = []
        act_function = self.get_torch_object_for_string(string_to_get=self.suggest_hyperparam_to_optuna('act_function'))
        in_channels = len(raw_data_functions.get_basic_aa_info()['A'])
        p = self.suggest_hyperparam_to_optuna('dropout')
        n_lin_layer = self.suggest_hyperparam_to_optuna('n_lin_layer')
        n_goal_latent_dim = 2 ** self.suggest_hyperparam_to_optuna('n_goal_latent_dim_exp')
        if n_lin_layer == 0:
            model.append(_torch_model.PackBlock())
        else:
            model.append(_torch_model.LinearPackEmbeddings(
                n_goal_dim_red=n_goal_latent_dim, act_function=act_function, in_channels=in_channels,
                dropout=p, n_layer=n_lin_layer)
            )
        lstm_in_channels = n_goal_latent_dim * n_lin_layer if n_lin_layer != 0 else in_channels
        hidden_size = 2**self.suggest_hyperparam_to_optuna('hidden_size_exp')
        model.append(
            _torch_model.LstmWithFlattenParams(
                input_size=lstm_in_channels, hidden_size=hidden_size,
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
            },
            'n_goal_latent_dim_exp': {
                'datatype': 'int',
                'lower_bound': 3,
                'upper_bound': 8
            },
            'n_lin_layer': {
                'datatype': 'int',
                'lower_bound': 0,
                'upper_bound': 3
            }
        }

