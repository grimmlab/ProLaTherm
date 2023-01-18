import torch

from . import _torch_model


class ProLaTherm(_torch_model.TorchModel):
    """
    Implementation of ProLaTherm

    See :obj:`~thermpred.model._base_model.BaseModel` and :obj:`~thermpred.model._torch_model.TorchModel` for more information on the attributes.
    """
    featureset = 'pretrained'

    def define_model(self) -> torch.nn.Sequential:
        model = []
        embedding_dim = 1024
        p = self.suggest_hyperparam_to_optuna('dropout')
        model.append(torch.nn.Dropout(p=p))
        model.append(_torch_model.AvgPoolTransformerOutput())
        out_features = int(0.5 * embedding_dim)
        model.append(torch.nn.Linear(in_features=embedding_dim, out_features=out_features))
        model.append(torch.nn.ReLU())
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
            'initial_units_factor': {
                'datatype': 'float',
                'lower_bound': 0.5,
                'upper_bound': 1,
                'step': 0.05
            }
        }

