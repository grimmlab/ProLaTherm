import torch

from . import _torch_model


class MlpEmbed(_torch_model.TorchModel):
    """
    Implementation of MLP with an embedding layer

    See :obj:`~thermpred.model._base_model.BaseModel` and :obj:`~thermpred.model._torch_model.TorchModel` for more information on the attributes.
    """
    featureset = 'sequence'

    def define_model(self) -> torch.nn.Sequential:
        """
        Definition of an LSTM network.

        Architecture:

            - tbd.

        """
        model = []
        embedding_dim = 1024
        model.append(torch.nn.Embedding(num_embeddings=self.size_alphabet+1, embedding_dim=embedding_dim))
        p = self.suggest_hyperparam_to_optuna('dropout')
        model.append(torch.nn.Dropout(p=p))
        model.append(_torch_model.AvgPoolTransformerOutput())
        if self.suggest_hyperparam_to_optuna('use_two_layer_classificiation_head'):
            out_features = int(0.5 * embedding_dim)
            model.append(torch.nn.Linear(in_features=embedding_dim, out_features=out_features))
            model.append(torch.nn.ReLU())
            model.append(torch.nn.BatchNorm1d(num_features=out_features))
            model.append(torch.nn.Dropout(p=p))
            model.append(torch.nn.Linear(in_features=out_features, out_features=self.n_outputs))
        else:
            model.append(torch.nn.Dropout(p=p))
            model.append(torch.nn.Linear(in_features=embedding_dim, out_features=self.n_outputs))

        return torch.nn.Sequential(*model)

    def define_hyperparams_to_tune(self) -> dict:
        """
        See :obj:`~thermpred.model._base_model.BaseModel` for more information on the format.

        See :obj:`~thermpred.model._torch_model.TorchModel` for more information on hyperparameters common for all torch models.
        """

        return {
            'use_two_layer_classificiation_head': {
                'datatype': 'categorical',
                'list_of_values': [False, True]
            },
            'initial_units_factor': {
                'datatype': 'float',
                'lower_bound': 0.5,
                'upper_bound': 1,
                'step': 0.05
            }
        }

