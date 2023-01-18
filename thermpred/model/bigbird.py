import torch

from . import _torch_model


class BigBird(_torch_model.TorchModel):
    """
    Implementation of BigBird

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
        n_t_blocks = self.suggest_hyperparam_to_optuna('n_transformer_blocks')
        fact_hidden_dim_mlp = 2  # self.suggest_hyperparam_to_optuna('factor_hidden_dim_mlp')
        block_size = self.suggest_hyperparam_to_optuna('block_size')
        num_global_tokens = self.suggest_hyperparam_to_optuna('factor_num_global_tokens') * block_size
        for t_block in range(n_t_blocks):
            model.append(
                _torch_model.BigBirdTransformerBlock(
                    latent_dim=transformer_dim, n_attention_heads=n_heads, dropout=p,
                    feedforward_hidden_dim_mult=fact_hidden_dim_mlp, block_size=block_size,
                    num_global_tokens=num_global_tokens)
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
                'list_of_values': [4, 8]
            },
            'factor_num_global_tokens': {
                'datatype': 'categorical',
                'list_of_values': [2, 3]
            },
            'block_size': {
                'datatype': 'categorical',
                'list_of_values': [8, 16, 32]
            }
        }

