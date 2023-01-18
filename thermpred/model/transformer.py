import torch

from . import _torch_model


class VanillaTransformer(_torch_model.TorchModel):
    """
    Implementation of a class for a vanilla Transformer

    See :obj:`~thermpred.model._base_model.BaseModel` and :obj:`~thermpred.model._torch_model.TorchModel` for more information on the attributes.
    """
    featureset = 'sequence'

    def define_model(self) -> torch.nn.Sequential:
        model = []
        embedding_dim = 2**self.suggest_hyperparam_to_optuna('embedding_dim_exp_transformer')
        n_heads = self.suggest_hyperparam_to_optuna('n_heads')
        transformer_dim = embedding_dim * n_heads
        p = self.suggest_hyperparam_to_optuna('dropout_transformer')
        model.append(_torch_model.AddSeqLengthInfoToX())
        model.append(
            _torch_model.TokenAndPositionalEmbedding(
                num_embeddings=self.size_alphabet+1, max_seq_length=self.max_len, embedding_dim=transformer_dim,
                dropout=p)
        )
        model.append(
            _torch_model.AvgPoolWithLengthInfo(
                kernel_size_avg_pool=self.suggest_hyperparam_to_optuna('kernel_size_avg_pool')
            )
        )
        n_t_blocks = self.suggest_hyperparam_to_optuna('n_transformer_blocks')
        fact_hidden_dim_mlp = self.suggest_hyperparam_to_optuna('factor_hidden_dim_mlp')
        for t_block in range(n_t_blocks):
            model.append(
                _torch_model.TransformerBlock(
                    k=transformer_dim, heads=n_heads, dropout=p, factor_hidden_dim_mlp=fact_hidden_dim_mlp)
            )
        model.append(_torch_model.DropSeqLengthInfo())
        model.append(_torch_model.AvgPoolTransformerOutput())
        model.append(torch.nn.Dropout(p=p))
        if self.suggest_hyperparam_to_optuna('use_two_layer_classificiation_head'):
            out_features = int(0.5 * transformer_dim)
            model.append(torch.nn.Linear(in_features=transformer_dim, out_features=out_features))
            model.append(torch.nn.ReLU())
            model.append(torch.nn.BatchNorm1d(num_features=out_features))
            model.append(torch.nn.Dropout(p=p))
            model.append(torch.nn.Linear(in_features=out_features, out_features=self.n_outputs))
        else:
            model.append(torch.nn.Dropout(p=p))
            model.append(torch.nn.Linear(in_features=transformer_dim, out_features=self.n_outputs))

        return torch.nn.Sequential(*model)

    def define_hyperparams_to_tune(self) -> dict:
        """
        See :obj:`~thermpred.model._base_model.BaseModel` for more information on the format.

        See :obj:`~thermpred.model._torch_model.TorchModel` for more information on hyperparameters common for all torch models.
        """

        return {
            'kernel_size_avg_pool': {
                'datatype': 'categorical',
                'list_of_values': [2, 3, 5]
            },
            'embedding_dim_exp_transformer': {
                'datatype': 'int',
                'lower_bound': 4,
                'upper_bound': 6
            },
            'use_two_layer_classificiation_head': {
                'datatype': 'categorical',
                'list_of_values': [False, True]
            },
            'n_transformer_blocks': {
                'datatype': 'int',
                'lower_bound': 2,
                'upper_bound': 6,
                'step': 2
            },
            'n_heads': {
                'datatype': 'int',
                'lower_bound': 2,
                'upper_bound': 6,
                'step': 2
            },
            'factor_hidden_dim_mlp': {
                'datatype': 'int',
                'lower_bound': 2,
                'upper_bound': 4
            },
            'weight_decay': {
                'datatype': 'categorical',
                'list_of_values': [1e-4, 1e-3, 1e-2, 1e-1]
            },
            'dropout_transformer': {
                'datatype': 'float',
                'lower_bound': 0.1,
                'upper_bound': 0.5,
                'step': 0.1
            },
            'label_smoothing': {
                'datatype': 'categorical',
                'list_of_values': [0.0, 0.1]
            },
            'learning_rate_transformer': {
                'datatype': 'categorical',
                'list_of_values': [1e-5, 1e-4, 1e-3]
            },
            'n_epochs_transformer': {
                'datatype': 'categorical',
                'list_of_values': [5000]  # [100, 200, 500, 1000, 5000]
            },
            'n_minibatches': {
                'datatype': 'categorical',
                'list_of_values': [2, 4, 8]
            }
        }

