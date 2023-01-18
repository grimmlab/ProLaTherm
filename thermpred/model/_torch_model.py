import abc
import numpy as np
import optuna
import torch.nn
import torch.utils.data
import copy
import math

from . import _base_model


class TorchModel(_base_model.BaseModel, abc.ABC):
    """
    Parent class based on :obj:`~thermpred.model._base_model.BaseModel` for all PyTorch models to share functionalities.
    See :obj:`~thermpred.model._base_model.BaseModel` for more information.

    *Attributes*

        *Inherited attributes*

        See :obj:`~thermpred.model._base_model.BaseModel`.

        *Additional attributes*

        - n_features (*int*): Number of input features to the model
        - size_alphabet (*int*): Number of unique nucleotides in sequence
        - batch_size (*int*): Batch size for batch-based training
        - n_epochs (*int*): Number of epochs for optimization
        - optimizer (*torch.optim.optimizer.Optimizer*): optimizer for model fitting
        - loss_fn: loss function for model fitting
        - early_stopping_patience (*int*): epochs without improvement before early stopping
        - early_stopping_point (*int*): epoch at which early stopping occured
        - device (*torch.device*): device to use, e.g. GPU

    :param task: ML task (regression or classification) depending on target variable
    :param optuna_trial: optuna.trial.Trial : trial of optuna for optimization
    :param n_outputs: Number of outputs of the model
    :param n_features: Number of input features to the model
    :param size_alphabet: Number of unique nucleotides in sequence
    :param batch_size: Batch size for batch-based training
    :param n_epochs: Number of epochs for optimization
    :param early_stopping_point: Stop training at defined epoch
    """

    def __init__(self, task: str, optuna_trial: optuna.trial.Trial, n_outputs: int = 1,
                 n_features: int = None, size_alphabet: int = None,
                 batch_size: int = None, n_epochs: int = None,
                 early_stopping_point: int = None, max_len: int = None, do_early_stopping: bool = True,
                 sorted_minibatches: bool = False, lr_reduce_points: list = None):
        self.sorted_minibatches = sorted_minibatches
        self.do_early_stopping = do_early_stopping
        self.all_hyperparams = self.common_hyperparams()  # add hyperparameters commonly optimized for all torch models
        self.n_features = n_features
        self.size_alphabet = size_alphabet  # relevant for models using embedding layers
        self.max_len = max_len
        super().__init__(task=task, optuna_trial=optuna_trial, n_outputs=n_outputs)
        if self.sorted_minibatches:
            self.batch_size = 1024
        else:
            self.batch_size = batch_size if batch_size is not None else self.suggest_hyperparam_to_optuna('batch_size')
        self.n_epochs = 5000
        learning_rate_to_use = 'learning_rate_transformer' if 'learning_rate_transformer' in self.all_hyperparams \
            else 'learning_rate'
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=self.suggest_hyperparam_to_optuna(learning_rate_to_use),
            weight_decay=self.suggest_hyperparam_to_optuna('weight_decay') if 'weight_decay' in self.all_hyperparams
            else 0
        )
        self.loss_fn = torch.nn.CrossEntropyLoss(
            label_smoothing=self.suggest_hyperparam_to_optuna('label_smoothing')
            if 'label_smoothing' in self.all_hyperparams else 0.0
        ) if task == 'classification' else torch.nn.MSELoss()
        # self.l1_factor = self.suggest_hyperparam_to_optuna('l1_factor')
        # early stopping if there is no improvement on validation loss for a certain number of epochs
        self.early_stopping_patience = self.suggest_hyperparam_to_optuna('early_stopping_patience') #if self.do_early_stopping else None
        self.early_stopping_point = early_stopping_point
        self.lr_reduce_points = lr_reduce_points
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.n_minibatches = self.suggest_hyperparam_to_optuna('n_minibatches') if sorted_minibatches else None

    def train_val_loop(self, X_train: np.array, y_train: np.array, X_val: np.array, y_val: np.array,
                       min_val_loss_imporovement: float = 1e-5) -> (np.array, np.array):
        """
        Implementation of a train and validation loop for  PyTorch models.
        See :obj:`~thermpred.model._base_model.BaseModel` for more information
        """
        train_loader = self.get_dataloader(X=X_train, y=y_train)
        val_loader = self.get_dataloader(X=X_val, y=y_val)
        best_model = copy.deepcopy(self.model)
        self.model.to(device=self.device)
        best_loss = None
        epochs_wo_improvement = 0
        scaler = torch.cuda.amp.GradScaler(enabled=False if self.device.type == 'cpu' else True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=20 if self.sorted_minibatches else 10, verbose=True, threshold_mode='abs',
            threshold=1e-5, factor=0.5, min_lr=1e-8
        )
        all_minibatches = []
        for inputs, targets in train_loader:
            all_minibatches.append(
                self.get_minibatches(inputs=inputs, targets=targets, n_minibatches=self.n_minibatches)
            )
        for epoch in range(self.n_epochs):
            # self.train_one_epoch(train_loader=train_loader, scaler=scaler, epoch=epoch)
            self.train_one_epoch(all_minibatches=all_minibatches, scaler=scaler, epoch=epoch,
                                 total_dataset_len=len(train_loader.dataset))
            val_loss = self.validate_one_epoch(val_loader=val_loader)
            if scheduler is not None:
                old_lr = self.optimizer.param_groups[0]['lr']
                scheduler.step(metrics=val_loss)
                if old_lr != self.optimizer.param_groups[0]['lr']:
                    if self.lr_reduce_points is None:
                        self.lr_reduce_points = []
                    self.lr_reduce_points.append(epoch)
            if best_loss is None or (val_loss + min_val_loss_imporovement) < best_loss:
                best_loss = val_loss
                epochs_wo_improvement = 0
                best_model = copy.deepcopy(self.model)
                #if self.do_early_stopping is False:
                #    self.early_stopping_point = epoch
            else:
                epochs_wo_improvement += 1
            print('Epoch ' + str(epoch + 1) + ' of ' + str(self.n_epochs))
            print('Current val_loss=' + str(val_loss) + ', best val_loss=' + str(best_loss))
            warmup = 10 if self.do_early_stopping else 100
            if epoch >= warmup and epochs_wo_improvement >= self.early_stopping_patience:
            #if self.do_early_stopping and epoch >= warmup and epochs_wo_improvement >= self.early_stopping_patience:
                print("Early Stopping at " + str(epoch + 1) + ' of ' + str(self.n_epochs))
                self.early_stopping_point = epoch - epochs_wo_improvement + 1
                self.model = best_model
                return self.predict(X_in=X_val)
        # set model to best_model (if early stopping did not apply)
        self.model = best_model
        return self.predict(X_in=X_val)

    def get_minibatches(self, inputs, targets, n_minibatches):
        minibatches = []
        block_size = self.optuna_trial.params['block_size'] if 'block_size' in self.optuna_trial.params else None
        if self.sorted_minibatches:
            num_minibatches = n_minibatches + 1
            input_lens = torch.Tensor((inputs.cpu().numpy()==0).argmax(axis=1))
            upper_lim_minibatches = [0]
            upper_lim_minibatches.extend(
                np.ceil(np.percentile(input_lens.numpy(), np.linspace(0, 100, num_minibatches)[1:])).astype(int))
            if block_size is not None:
                for ind in range(1, len(upper_lim_minibatches)):
                    upper_lim_minibatches[ind] = math.ceil(upper_lim_minibatches[ind]/block_size) * block_size
                if upper_lim_minibatches[-1] > self.max_len:
                    pad = torch.zeros(self.batch_size, math.ceil(self.max_len / block_size) * block_size)
                    pad[:, :self.max_len] = inputs
                    inputs = pad
            for lower, upper in zip(range(0, num_minibatches-1), range(1, num_minibatches)):
                indices = np.where(
                    (upper_lim_minibatches[lower] < input_lens) & (input_lens <= upper_lim_minibatches[upper]),
                    range(0, len(input_lens)), -1)
                indices = np.delete(indices, np.where(indices == -1))
                minibatches.append((inputs[indices, :upper_lim_minibatches[upper]], targets[indices]))
        else:
            minibatches = [(inputs, targets)]
        return minibatches

    def train_one_epoch(self, all_minibatches: list, scaler, epoch, total_dataset_len):
        """
        Train one epoch

        :param train_loader: DataLoader with training data
        """
        self.model.train()
        total_loss = 0
        for minibatches in all_minibatches:
            for minibatch in minibatches:
                inp, targ = minibatch[0].to(device=self.device), minibatch[1].to(device=self.device)
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type):
                    outputs = self.model(inp)
                    loss = self.get_loss(outputs=outputs, targets=targ)
                    total_loss += loss
                # l1_loss = 0
                # for param in self.model.parameters():
                #     l1_loss += torch.sum(torch.abs(param))
                # loss += self.l1_factor * l1_loss
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

    def validate_one_epoch(self, val_loader: torch.utils.data.DataLoader) -> float:
        """
        Validate one epoch

        :param val_loader: DataLoader with validation data

        :return: loss based on loss-criterion
        """
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device=self.device), targets.to(device=self.device)
                with torch.autocast(device_type=self.device.type):
                    outputs = self.model(inputs)
                    total_loss += self.get_loss(outputs=outputs, targets=targets).item()
        return total_loss / len(val_loader.dataset)

    def retrain(self, X_retrain: np.array, y_retrain: np.array):
        """
        Implementation of the retraining for PyTorch models.
        See :obj:`~thermpred.model._base_model.BaseModel` for more information
        """
        retrain_loader = self.get_dataloader(X=X_retrain, y=y_retrain)
        n_epochs_to_retrain = self.n_epochs if self.early_stopping_point is None else self.early_stopping_point
        self.model.to(device=self.device)
        scaler = torch.cuda.amp.GradScaler(enabled=False if self.device.type == 'cpu' else True)
        all_minibatches = []
        for inputs, targets in retrain_loader:
            all_minibatches.append(
                self.get_minibatches(inputs=inputs, targets=targets, n_minibatches=self.n_minibatches)
            )
        for epoch in range(n_epochs_to_retrain):
            print('Retrain: Epoch ' + str(epoch + 1) + ' of ' + str(n_epochs_to_retrain))
            # self.train_one_epoch(retrain_loader, scaler=scaler, epoch=epoch)
            self.train_one_epoch(all_minibatches=all_minibatches, scaler=scaler, epoch=epoch,
                                 total_dataset_len=len(retrain_loader.dataset))
            if self.lr_reduce_points is not None:
                if epoch in self.lr_reduce_points:
                    print('Reduced lr from ' + str(self.optimizer.param_groups[0]['lr']) +
                          ' to ' + str(max(0.5 * self.optimizer.param_groups[0]['lr'], 1e-8)))
                    self.optimizer.param_groups[0]['lr'] = max(0.5 * self.optimizer.param_groups[0]['lr'], 1e-8)

    def predict(self, X_in: np.array) -> (np.array, np.array):
        """
        Implementation of a prediction based on input features for PyTorch models.
        See :obj:`~thermpred.model._base_model.BaseModel` for more information
        """
        dataloader = self.get_dataloader(X=X_in, shuffle=False)
        self.model.eval()
        predictions = None
        with torch.no_grad():
            for inputs in dataloader:
                inputs = inputs.to(device=self.device)
                with torch.autocast(device_type=self.device.type):
                    outputs = self.model(inputs)
                predictions = torch.clone(outputs) if predictions is None else torch.cat((predictions, outputs))
        if self.task == 'classification':
            scores = torch.nn.functional.softmax(predictions, dim=1)[:, 1]
            _, predictions = torch.max(predictions, 1)
        return predictions.cpu().detach().numpy(), scores.double().cpu().detach().numpy()

    def get_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss based on the outputs and targets

        :param outputs: outputs of the model
        :param targets: targets of the dataset

        :return: loss
        """
        if type(self.loss_fn) in [torch.nn.CrossEntropyLoss, torch.nn.NLLLoss]:
            targets = targets.long()
        return self.loss_fn(outputs, targets)

    def get_dataloader(self, X: np.array, y: np.array = None, shuffle: bool = True) -> torch.utils.data.DataLoader:
        """
        Get a Pytorch DataLoader using the specified data and batch size

        :param X: feature matrix to use
        :param y: optional target vector to use
        :param shuffle: shuffle parameter for DataLoader

        :return: Pytorch DataLoader
        """
        # drop last sample if last batch would only contain one sample
        if (len(X) % self.batch_size) == 1:
            X = X[:-1]
            y = y[:-1] if y is not None else None
        if self.featureset == 'sequence':
            X = torch.from_numpy(X).long()
        #elif self.featureset == 'pretrained':
        #    X = torch.from_numpy(X)
        else:
            X = torch.from_numpy(X).float()
        if self.featureset == 'aa_desc_matrix':
            # Adapt to PyTorch ordering for CNN (BATCH_SIZE, CHANNELS, SIGNAL)
            X = torch.swapaxes(X, 1, 2)
        y = torch.reshape(torch.from_numpy(y).float(), (-1, 1)) if y is not None else None
        y = torch.flatten(y) if (self.task == 'classification' and y is not None) else y
        dataset = torch.utils.data.TensorDataset(X, y) if y is not None \
            else X
        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=shuffle,
                                           num_workers=0, pin_memory=True)

    def common_hyperparams(self):
        """
        Add hyperparameters that are common for PyTorch models.
        Do not need to be included in optimization for every child model.
        Also See :obj:`~thermpred.model._base_model.BaseModel` for more information
        """
        if self.do_early_stopping:
            early_stopping_patience = {
                'datatype': 'int',
                'lower_bound': 0,
                'upper_bound': 10,
                'step': 5
            }
        else:
            early_stopping_patience = {
                'datatype': 'categorical',
                'list_of_values': [50]
            }

        return {
            'dropout': {
                'datatype': 'float',
                'lower_bound': 0,
                'upper_bound': 0.5,
                'step': 0.1
            },
            'act_function': {
                'datatype': 'categorical',
                'list_of_values': ['relu', 'tanh']
            },
            'batch_size': {
                'datatype': 'categorical',
                'list_of_values': [64, 128, 256]
            },
            'n_epochs': {
                'datatype': 'categorical',
                'list_of_values': [100, 500, 1000, 5000, 10000]
            },
            'learning_rate': {
                'datatype': 'categorical',
                'list_of_values': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
            },
            'early_stopping_patience': early_stopping_patience,
            'l1_factor': {
                'datatype': 'float',
                'lower_bound': 0,
                'upper_bound': 10**3
            },
            'embedding_dim_exp': {
                'datatype': 'int',
                'lower_bound': 4,
                'upper_bound': 10
            }
        }

    @staticmethod
    def get_torch_object_for_string(string_to_get: str):
        """
        Get the torch object for a specific string, e.g. when suggesting to optuna as hyperparameter

        :param string_to_get: string to retrieve the torch object

        :return: torch object
        """
        string_to_object_dict = {
            'relu': torch.nn.ReLU(),
            'tanh': torch.nn.Tanh()
        }
        return string_to_object_dict[string_to_get]


class LstmWithFlattenParams(torch.nn.Module):
    """Regular LSTM with parameterizable number of layers, input_size, hidden_size and bidirectional"""

    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                  dropout=dropout, batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        self.lstm.flatten_parameters()  # for more efficient memory usage
        out = self.lstm(x)
        return out


class ExtractTensor(torch.nn.Module):
    """Extract last hidden state from LSTM output for classification, either bidirectional or regular"""

    def __init__(self, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional

    def forward(self, x):
        packed_output, (hn, cn) = x  # unpack output
        if self.bidirectional:
            hn_return = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)  # concat forward and backward path
        else:
            hn_return = hn[-1, :, :]
        return hn_return


class EmbedAndPackBlock(torch.nn.Module):
    """Get embeddings for a sequence and pack padded sequences for efficient processing with RNNs"""

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding_layer = \
            torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=0)

    def forward(self, x):
        x_embedded = self.embedding_layer(x)
        x_lens = torch.Tensor((x.cpu().numpy()==0).argmax(axis=1))
        x_lens[x_lens == 0] = x.shape[1]  # did not find any 0 -> full sequence length
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(
            x_embedded, x_lens, batch_first=True, enforce_sorted=False
        )
        return x_packed


class PackBlock(torch.nn.Module):
    """Pack inputs with sequence length info for efficient processing with RNNs"""

    def forward(self, x):
        x_lens = torch.Tensor((x[:, 0, :].cpu().numpy()==0).argmax(axis=1))
        x_lens[x_lens == 0] = x.shape[1]  # did not find any 0 -> full sequence length
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(
            torch.swapaxes(x, 1, 2), x_lens, batch_first=True, enforce_sorted=False
        )
        return x_packed


class LinearPackEmbeddings(torch.nn.Module):
    """Reduce latent dim of embeddings with a feedforward network
    and pack padded sequences for efficient processing with RNNs """

    def __init__(self, act_function, in_channels, n_goal_dim_red, dropout, n_layer=1):
        super().__init__()
        self.lin_layers = torch.nn.ModuleList()
        self.act_layers = torch.nn.ModuleList()
        self.bn_layers = torch.nn.ModuleList()
        self.dropout_layers = torch.nn.ModuleList()
        in_features = in_channels
        for layer in range(n_layer):
            self.lin_layers.append(torch.nn.Linear(in_features=in_features, out_features=n_goal_dim_red * (layer+1)))
            self.act_layers.append(act_function)
            self.dropout_layers.append(torch.nn.Dropout(dropout))
            in_features = n_goal_dim_red * (layer+1)

    def forward(self, x):
        if self.lin_layers[0].in_features != x.shape[2]:
            x = torch.swapaxes(x, 1, 2)
        x_lens = torch.from_numpy((x.cpu().numpy()==np.zeros((x.shape[1], x.shape[2]))).argmax(axis=1)[:, 0])
        x_lens[x_lens == 0] = x.shape[1]  # did not find any 0 -> full sequence length
        for layer in range(len(self.lin_layers)):
            x = self.dropout_layers[layer](self.act_layers[layer](self.lin_layers[layer](x)))
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, x_lens, batch_first=True, enforce_sorted=False
        )
        return x_packed


class AvgPoolWithLengthInfo(torch.nn.Module):
    """Do average pooling and add sequence length information to output"""

    def __init__(self, kernel_size_avg_pool):
        super().__init__()
        self.avgpool = torch.nn.AvgPool1d(kernel_size=kernel_size_avg_pool, stride=kernel_size_avg_pool)

    def forward(self, x):
        x_avg = torch.swapaxes(self.avgpool(torch.swapaxes(x[0], 1, 2)), 1, 2)
        x_lens = torch.clamp(torch.ceil(x[1] / self.avgpool.kernel_size[0]), max=x_avg.shape[1]).to(torch.int16)
        return x_avg, x_lens


class SwapAxes12(torch.nn.Module):
    """Swap axes, sometimes needed as convolutional and recurrent layers work with different order"""
    def forward(self, x):
        return torch.swapaxes(x, 1, 2)


class SelfAttentionSplitted(torch.nn.Module):
    """
    Self attention block, efficient version with split along latent dim
    """

    def __init__(self, k, heads=8):
        super().__init__()
        self.heads = heads
        self.head_dim = k // heads
        # queries, keys and values for all heads (as a single concatenated vector to enable single matrix mult.)
        # input dim k: latent dim; output dim k*heads: concatenated outputs for all heads
        self.tokeys = torch.nn.Linear(k, k, bias=False)
        self.toqueries = torch.nn.Linear(k, k, bias=False)
        self.tovalues = torch.nn.Linear(k, k, bias=False)

        # unification of outputs of all heads -> input dim heads*k -> output dim k (equal to input size latent dim)
        self.unifyheads = torch.nn.Linear(k, k)

    def forward(self, x):
        attn_penalty = -10000
        # batch size b, sequence length t, latent dim k
        b, t, k = x[0].size()

        # output linear module: (b, t, h*k) -> reshape to (b, t, h, k) to split the heads into separate dims
        queries = self.toqueries(x[0]).view(b, t, self.heads, self.head_dim)
        keys = self.tokeys(x[0]).view(b, t, self.heads, self.head_dim)
        values = self.tovalues(x[0]).view(b, t, self.heads, self.head_dim)

        # transpose: (b, t, h, k) -> (b, h, t, k) --> fold heads into batch dim to enable torch.bmm (b*h, t, k)
        keys = keys.transpose(1, 2).contiguous().view(b * self.heads, t, self.head_dim)
        queries = queries.transpose(1, 2).contiguous().view(b * self.heads, t, self.head_dim)
        values = values.transpose(1, 2).contiguous().view(b * self.heads, t, self.head_dim)

        # scale keys and queries by sqrt_4(k) as softmax can be sensitive to large values
        # scaling by sqrt(k) changed to sqrt_4(k) as it is moved before the dot product (to save memory for larges seqs)
        queries = queries / (self.head_dim ** (1 / 4))
        keys = keys / (self.head_dim ** (1 / 4))

        # dot product of queries and keys using bmm --> result is of shape (b*h, t, t) ((b*h, t, k) * (b*h, k, t))
        dot = torch.bmm(queries, keys.transpose(1, 2))
        # mask = torch.zeros((b*self.heads, t))
        # for seq in range(b):
        #     mask[seq::self.heads, :seq_lenghts[seq]] = 1
        # mask = mask.to(device=dot.device)
        # dot += (1. - mask.view(b*h, 1, t)) * attn_penalty
        dot = dot.view(b, self.heads, t, t)
        mask = torch.zeros((b, t), dtype=torch.bfloat16)
        for seq in range(b):
            mask[seq, :x[1][seq]] = 1
        mask = mask.to(device=dot.device)
        dot += (1. - mask.view(b, 1, 1, t)) * attn_penalty
        dot = dot.view(b * self.heads, t, t)
        # row-wise normalization using softmax
        dot = torch.nn.functional.softmax(dot, dim=2)

        # apply the self attention to values: (b*h, t, t) * (b*h, t, k) --> (b*h, t, k) --> reshape to separate heads
        out = torch.bmm(dot, values).view(b, self.heads, t, self.head_dim)

        # dim swap: (b, h, t, k) -> (b, t, h, k) -> reshape to (b, t, h*k)
        out = out.transpose(1, 2).contiguous().view(b, t, self.heads * self.head_dim)
        # unify heads: input shape (b, t, h*k) -> output shape (b, t, k)
        return self.unifyheads(out)


class TransformerBlock(torch.nn.Module):
    """
    Transformer Block with pre-layer-norm:
        layer-norm -> (self-att + skip-connection)-> dropout -> layer-norm -> (mlp + skip-connection) -> dropout

    """
    def __init__(self, k, heads, dropout, factor_hidden_dim_mlp: int = 2):
        super().__init__()
        # self.attention = SelfAttention(k, heads=heads)
        self.attention = SelfAttentionSplitted(k, heads=heads)

        self.norm1 = torch.nn.LayerNorm(k)
        self.norm2 = torch.nn.LayerNorm(k)
        self.init_layer_norm()

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(k, factor_hidden_dim_mlp * k),
            torch.nn.GELU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(factor_hidden_dim_mlp * k, k)
        )
        self.dropout = torch.nn.Dropout(p=dropout)

    def init_layer_norm(self):
        for layer_norm in [self.norm1, self.norm2]:
            bound = 1 / (layer_norm.weight.size()[0] ** 1 / 2)
            torch.nn.init.uniform_(layer_norm.weight, -bound, bound)

    def forward(self, x):
        # first norm, then attend and add with skip-connection
        x_ = x[0] + self.dropout(self.attention((self.norm1(x[0]), x[1])))
        # first norm, then mlp and add with skip-connection
        x_ = x_ + self.dropout(self.mlp(self.norm2(x_)))

        return x_, x[1]


class TokenAndPositionalEmbedding(torch.nn.Module):
    """Both tokens and positional information are learned via embeddings, added and processed with a dropout layer"""

    def __init__(self, num_embeddings, embedding_dim, max_seq_length, dropout):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=0
        )
        self.pos_embedding = torch.nn.Embedding(
            num_embeddings=max_seq_length, embedding_dim=embedding_dim, padding_idx=0
        )
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        tokens = self.token_embedding(x[0])
        b, t, k = tokens.size()

        positions = torch.arange(t).to(device=x[0].device)
        positions = self.pos_embedding(positions)[None, :, :].expand(b, t, k)
        x_add = tokens + positions

        return self.dropout(x_add), x[1]


class TokenAndPositionalEncoding(torch.nn.Module):
    """Tokens are learned with an embedding layer.
    Positional information is encoded with commonly-used sine/cosine mechanism.
    Both are added (with scaling tokens) and processed with a dropout layer"""

    def __init__(self, num_embeddings, embedding_dim, max_seq_length, dropout):
        super().__init__()
        self.max_len = max_seq_length
        self.token_embedding = torch.nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=0
        )
        self.pos_encoding = PositionalEncoding(d_model=embedding_dim, max_len=max_seq_length)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        device = x[0].device
        x_vals = x[0]
        seq_lenghts = x[1]
        tokens = self.token_embedding(x_vals.to(torch.int32))
        positions = self.pos_encoding(x_vals)
        x_add = (tokens * (self.token_embedding.embedding_dim ** 1/2)) + positions[:, :tokens.size(1)]

        return self.dropout(x_add.to(device)), seq_lenghts


class PositionalEncoding(torch.nn.Module):
    """Positional encoding with sine and cosine information"""

    def __init__(self, d_model: int, max_len):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1), :].to(device=x.device)


class AddSeqLengthInfoToX(torch.nn.Module):
    """Add sequence length to output"""

    def forward(self, x):
        x_lens = torch.Tensor((x.cpu().numpy()==0).argmax(axis=1)).to(torch.int32)
        return x, x_lens.to(x.device)


class DropSeqLengthInfo(torch.nn.Module):
    """Drop Sequence length for standard components"""
    def forward(self, x):
        return x[0]


class AvgPoolTransformerOutput(torch.nn.Module):
    """Average transformer output along sequence-dim to process with linear head classifier"""
    def forward(self, x):
        return x.mean(dim=1)


class BigBirdTransformerBlock(torch.nn.Module):
    """
    Transformer block with pre-norm and big bird self-attention
    """
    def __init__(self, latent_dim, n_attention_heads, block_size, num_global_tokens, feedforward_hidden_dim_mult=2, dropout=0.0):
        super().__init__()

        self.attention = BigBirdBlockSparseSelfAttention(
            latent_dim=latent_dim,
            num_attention_heads=n_attention_heads,
            block_size=block_size
        )
        self.num_global_tokens = num_global_tokens

        # normalization before the self-attention
        self.norm1 = torch.nn.LayerNorm(latent_dim)
        # normalization before the MLP
        self.norm2 = torch.nn.LayerNorm(latent_dim)
        self.init_layer_norm()

        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, feedforward_hidden_dim_mult * latent_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(feedforward_hidden_dim_mult * latent_dim, latent_dim)
        )

        self.dropout = torch.nn.Dropout(p=dropout)

    def init_layer_norm(self):
        for layer_norm in [self.norm1, self.norm2]:
            bound = 1 / (layer_norm.weight.size()[0] ** 1 / 2)
            torch.nn.init.uniform_(layer_norm.weight, -bound, bound)

    @staticmethod
    def create_attention_mask(seq_lengths, b, t):
        mask = torch.zeros((b, t))
        for seq in range(b):
            mask[seq, :seq_lengths[seq]] = 1
        mask = mask.to(device=seq_lengths.device)
        return mask

    def forward(self, x, band_mask=None, from_mask=None, to_mask=None, attention_bias=None):
        seq_lengths, x = x[1], x[0]
        blocked_mask, inner_band_mask, from_mask, to_mask = \
            self.attention.create_masks(
                attention_mask=self.create_attention_mask(seq_lengths, b=x.size()[0], t=x.size()[1]),
                block_size=self.attention.block_size,
                n_global_tokens=self.num_global_tokens, to_device=x.device)
        attention = self.attention(
            tokens=self.norm1(x),
            num_global_tokens=self.num_global_tokens,
            band_mask=inner_band_mask,
            from_mask=from_mask,
            to_mask=to_mask,
            attention_bias=attention_bias
        )
        x = attention + x

        x_mlp = self.norm2(x)
        x_mlp = self.dropout(x_mlp)
        x_mlp = self.feedforward(x_mlp)

        x = x_mlp + x

        x = self.dropout(x)

        return x, seq_lengths


class BigBirdBlockSparseSelfAttention(torch.nn.Module):
    """
    Inspired by but different to
    https://github.com/vasudevgupta7/transformers/blob/5f2d6a0c93ca2017961199aa04a344b9b779d454/src/transformers/models/big_bird/modeling_big_bird.py
    """
    def __init__(self, latent_dim: int, num_attention_heads: int, block_size: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_attention_heads = num_attention_heads
        self.block_size = block_size

        if latent_dim % num_attention_heads != 0:
            raise ValueError(f"Latent dimension ({latent_dim}) is not a multiple"
                             f" of the number of attention heads ({num_attention_heads})")

        self.attention_head_size = latent_dim // num_attention_heads

        self.to_queries_linear = torch.nn.Linear(in_features=latent_dim, out_features=latent_dim, bias=False)
        self.to_keys_linear = torch.nn.Linear(in_features=latent_dim, out_features=latent_dim, bias=False)
        self.to_values_linear = torch.nn.Linear(in_features=latent_dim, out_features=latent_dim, bias=False)

        self.unify_heads_linear = torch.nn.Linear(self.latent_dim, self.latent_dim)

    def transpose_for_scores(self, x):
        """
        x of shape (batch, seq len, dim) is split and transformed into (batch, num heads, seq len, head dim)
        for easier transformation.
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, tokens, num_global_tokens: int, band_mask=None, from_mask=None, to_mask=None, attention_bias=None):
        """
        Parameters
        ----------
        tokens: Tensor of shape (batch_size, seqlen, latent dim), input to Attention block
        num_global_tokens: Number of global tokens which are expected at the top of the tokens, i.e. tokens[:num_global_tokens]
            are global attention tokens.
        band_mask, from_mask, to_mask: Masks as returned from `create_masks`
        attention_bias: Bias in shape (batch, num_heads, sequence_len, sequence_len)
        """

        batch_size, seqlen, _ = tokens.size()
        block_size = self.block_size

        assert seqlen % block_size == 0, "Sequence length must be multiple of block size"
        assert num_global_tokens % block_size == 0, "Number of global tokens must be multiple of block size"

        # Transform all tokens to queries, keys, and values
        # All have shape [batch_size, num_heads, seqlen, head_dim]
        query_layer = self.transpose_for_scores(self.to_queries_linear(tokens))
        key_layer = self.transpose_for_scores(self.to_keys_linear(tokens))
        value_layer = self.transpose_for_scores(self.to_values_linear(tokens))

        # Define some shorthands
        heads = self.num_attention_heads
        rsqrt_d = 1. / (self.attention_head_size ** 0.5)
        batch = batch_size
        block = block_size
        seq = seqlen
        dim = self.attention_head_size
        n_blocks = seq // block
        n_global_blocks = num_global_tokens // block  # number of tokens with global blocks

        # Transform queries, keys and values into block form [batch, heads, n_blocks, block, dim]
        blocked_query_matrix = query_layer.view(batch, heads, n_blocks, block, dim)
        blocked_key_matrix = key_layer.view(batch, heads, n_blocks, block, dim)
        blocked_value_matrix = value_layer.view(batch, heads, n_blocks, block, dim)

        attn_mask_penalty = -10000.0

        # --- Step 1: Queries are all global tokens, keys are all keys. Multiply all global tokens with all keys.
        # [batch, heads, num_global_tokens, dim] x [batch, heads, seqlen, dim]
        # ==> [batch, heads, num_global_tokens, seqlen]
        first_product = self.torch_bmm_nd_transpose(query_layer[:, :, :num_global_tokens, :], key_layer, ndim=4)

        first_product = first_product * rsqrt_d
        if attention_bias is not None:
            first_product += attention_bias[:, :, :num_global_tokens, :]
        first_product += (1.0 - to_mask) * attn_mask_penalty
        first_attn_weights = torch.nn.functional.softmax(first_product, dim=-1)  # [batch, heads, num_global_tokens, seqlen]

        # [batch, heads, num_global_tokens, seqlen] x [batch, heads, seqlen, head_dim]
        # ==> [batch, heads, num_global_tokens, dim]
        first_context_layer = self.torch_bmm_nd(first_attn_weights, value_layer, ndim=4)
        # Put into block form for later concatenation
        first_context_layer = first_context_layer.view(batch, heads, n_global_blocks, block, dim)

        # -- Step 2: First block after global blocks x (sliding keys, global keys)
        second_context_layer = None
        if n_blocks - n_global_blocks >= 1:
            # Take two neighboring blocks of first block if possible
            q1_num_blocks = min(3, n_blocks-n_global_blocks)

            # start index of first block after global blocks
            second_block_start_idx = n_global_blocks * block

            second_key_mat = torch.cat([
                key_layer[:, :, :num_global_tokens],
                key_layer[:, :, second_block_start_idx: second_block_start_idx + (q1_num_blocks * block)]

            ], dim=2)  # [batch, heads, block * (n_global_blocks + q1_num_blocks), dim]

            second_value_mat = torch.cat([
                value_layer[:, :, :num_global_tokens],
                value_layer[:, :, second_block_start_idx: second_block_start_idx + (q1_num_blocks * block)]

            ], dim=2)  # [batch, heads, block * (n_global_blocks + q1_num_blocks), dim]

            # [batch, heads, block, dim] x [batch, heads, block * (n_global_blocks + q1_num_blocks), dim]
            # ==> [batch, heads, block, block * (n_global_blocks + q1_num_blocks)]
            second_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, n_global_blocks], second_key_mat, ndim=4)

            # Mask is all global tokens and taken neighboring tokens
            second_seq_mask = to_mask[:, :, :, :block * (n_global_blocks + q1_num_blocks)]

            second_product = second_product * rsqrt_d
            if attention_bias is not None:
                second_attention_bias = torch.cat([
                    attention_bias[:, :, second_block_start_idx: second_block_start_idx + block, :num_global_tokens],
                    attention_bias[:, :, second_block_start_idx: second_block_start_idx + block, second_block_start_idx: second_block_start_idx + (q1_num_blocks * block)]
                    ], dim=3)
                second_product += second_attention_bias
            second_product += (1.0 - second_seq_mask) * attn_mask_penalty
            second_attn_weights = torch.nn.functional.softmax(second_product, dim=-1)  # [batch, heads, block, block * (n_global_blocks + q1_num_blocks)]

            # [batch, heads, block, block * (n_global_blocks + q1_num_blocks)] x [batch, heads, block * (n_global_blocks + q1_num_blocks), dim]
            # ==> [batch, heads, block, dim]
            second_context_layer = self.torch_bmm_nd(second_attn_weights, second_value_mat, ndim=4)

            second_context_layer.unsqueeze_(dim=2)  # [batch, heads, 1, block, dim], for later concat

        # -- Step 3: Second block to next-to-last after global blocks x (sliding keys, global keys)
        n_middle_blocks = n_blocks - n_global_blocks - 2  # - 2 for first and last non-global block
        context_layer = None
        if n_middle_blocks >= 1:  # we have at least one middle block at position (n_global_blocks + 1), which is not the last block
            # copy and shift the key matrix
            exp_blocked_key_matrix = torch.cat([
                blocked_key_matrix[:, :, n_global_blocks: n_global_blocks + n_middle_blocks],  # starting from first non-global block
                blocked_key_matrix[:, :, n_global_blocks + 1: n_global_blocks + n_middle_blocks + 1],
                blocked_key_matrix[:, :, n_global_blocks + 2: n_global_blocks + n_middle_blocks + 2],
            ], dim=3)  # [batch, head, n_middle_blocks, 3*block, dim]

            exp_blocked_value_matrix = torch.cat([
                blocked_value_matrix[:, :, n_global_blocks: n_global_blocks + n_middle_blocks],
                blocked_value_matrix[:, :, n_global_blocks + 1: n_global_blocks + n_middle_blocks + 1],
                blocked_value_matrix[:, :, n_global_blocks + 2: n_global_blocks + n_middle_blocks + 2],
            ], dim=3)  # [batch, head, n_middle_blocks, 3*block, dim]

            # [batch, head, n_middle_blocks, block, dim]
            middle_query_matrix = blocked_query_matrix[:, :, n_global_blocks + 1: n_global_blocks + 1 + n_middle_blocks]

            # Sliding attention scores
            # [batch, head, n_middle_blocks, block, dim] x [batch, head, n_middle_blocks, 3*block, dim]
            # ==> [batch, head, n_middle_blocks, block, 3*block]
            inner_band_product = self.torch_bmm_nd_transpose(middle_query_matrix, exp_blocked_key_matrix, ndim=5)
            inner_band_product = inner_band_product * rsqrt_d

            # Add attention bias for inner band
            middle_query_from_idx = (n_global_blocks + 1) * block
            middle_query_to_idx = (n_global_blocks + 1 + n_middle_blocks) * block
            if attention_bias is not None:
                # This is not very effective. Need to find better way
                inner_band_attention_bias = torch.cat([
                    attention_bias[:, :, middle_query_from_idx + i * block: middle_query_from_idx + (i+1)*block,
                                    middle_query_from_idx + (i-1) * block: middle_query_from_idx + (i+2) * block]
                    for i in range(n_middle_blocks)
                ], dim=3).view(batch, heads, n_middle_blocks, block, 3*block)

                inner_band_product += inner_band_attention_bias

            # First blocks are global
            # [batch, head, n_middle_blocks, block, dim] x [batch, head, n_global_tokens, dim]
            # ==> [batch, head, n_middle_blocks, block, n_global_tokens
            first_band_product = torch.einsum(
                "bhlqd,bhkd->bhlqk", middle_query_matrix, key_layer[:, :, :num_global_tokens]
            )
            first_band_product = first_band_product * rsqrt_d
            if attention_bias is not None:
                first_band_attention_bias = attention_bias[:, :, middle_query_from_idx: middle_query_to_idx, :num_global_tokens]
                first_band_attention_bias = first_band_attention_bias.view(batch, heads, n_middle_blocks, block, num_global_tokens)
                first_band_product += first_band_attention_bias

            # masking padded tokens
            first_band_product += (1.0 - to_mask[:, :, :, :num_global_tokens].unsqueeze(3)) * attn_mask_penalty
            inner_band_product += (1.0 - band_mask) * attn_mask_penalty

            # completing attention scores for middle blocks
            # [batch, heads, n_middle_blocks, block, n_global_tokens + 3*block]
            band_product = torch.cat([first_band_product, inner_band_product], dim=-1)

            # safely doing softmax since attention matrix is completed
            attn_weights = torch.nn.functional.softmax(band_product, dim=-1)  # [batch, heads, n_middle_blocks, block, n_global_tokens + 3*block]

            # contribution of sliding keys
            # [batch, heads, n_middle_blocks, block, 3*block] x [batch, head, n_middle_blocks, 3*block, dim]
            # ==> [batch, head, n_middle_blocks, block, dim]
            context_layer = self.torch_bmm_nd(attn_weights[:, :, :, :, num_global_tokens:], exp_blocked_value_matrix, ndim=5)

            # adding contribution of global keys
            # [batch, heads, n_middle_blocks, block, num_global_tokens] x [batch, head, n_global_tokens, dim]
            # ==> [batch, heads, n_middle_blocks, block, dim]
            context_layer += torch.einsum(
                "bhlqk,bhkd->bhlqd", attn_weights[:, :, :, :, :num_global_tokens], value_layer[:, :, :num_global_tokens]
            )

        # -- Step 3: Last block x (sliding keys, global keys)
        # check if there is a last block which is not the first non-global block
        last_context_layer = None
        if n_blocks - n_global_blocks - 1 >= 1:
            ql_num_blocks = min(3, n_blocks - n_global_blocks)

            last_key_mat = torch.cat([
                key_layer[:, :, :num_global_tokens],
                key_layer[:, :, -(ql_num_blocks * block):]
            ], dim=2)  # [batch, heads, block * (n_global_blocks + ql_num_blocks), dim]

            last_value_mat = torch.cat([
                value_layer[:, :, :num_global_tokens],
                value_layer[:, :, -(ql_num_blocks * block):]
            ], dim=2)  # [batch, heads, block * (n_global_blocks + ql_num_blocks), dim]

            # [batch, head, block, dim] x [batch, heads, block * (n_global_blocks + ql_num_blocks), dim]
            # ==> [batch, head, block, block * (n_global_blocks + ql_num_blocks)]
            last_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, -1], last_key_mat, ndim=4)

            # Mask is all global tokens and the neighboring tokens of last block
            last_seq_mask = torch.cat([
                to_mask[:, :, :, :num_global_tokens],
                to_mask[:, :, :, -(ql_num_blocks * block):]
            ], dim=3)

            last_product = last_product * rsqrt_d
            if attention_bias is not None:
                last_attention_bias = torch.cat([
                    attention_bias[:, :, -block:, :num_global_tokens],
                    attention_bias[:, :, -block:, -(ql_num_blocks * block):],
                ], dim=3)
                last_product += last_attention_bias
            last_product += (1.0 - last_seq_mask) * attn_mask_penalty
            last_attn_weights = torch.nn.functional.softmax(
                last_product,
                dim=-1)  # [batch, heads, block, block * (n_global_blocks + ql_num_blocks)]

            # [batch, heads, block, block * (n_global_blocks + ql_num_blocks)] x [batch, heads, block * (n_global_blocks + ql_num_blocks), dim]
            # ==> [batch, heads, block, dim]
            last_context_layer = self.torch_bmm_nd(last_attn_weights, last_value_mat, ndim=4)
            last_context_layer.unsqueeze_(dim=2)  # [batch, heads, 1, block, dim], for later concat

        # Concatenate all context layers to have outputs for all tokens, and the reshape back from block form to
        # regular form
        layers_to_cat = [first_context_layer]
        if second_context_layer is not None:
            layers_to_cat.append(second_context_layer)
        if context_layer is not None:
            layers_to_cat.append(context_layer)
        if last_context_layer is not None:
            layers_to_cat.append(last_context_layer)

        context_layer = torch.cat(layers_to_cat, dim=2)
        context_layer = context_layer.view((batch, heads, seqlen, dim)) * from_mask  # zero out all padded tokens

        # transpose back to shape (batch, seqlen, heads * dim) and unify the heads
        out = context_layer.transpose(1, 2).contiguous().view(batch, seqlen, self.latent_dim)

        return self.unify_heads_linear(out)

    @staticmethod
    def create_masks(attention_mask: torch.Tensor, block_size: int, n_global_tokens: int, to_device: torch.device = None):
        """
        `attention_mask` is expected in shape (batch_size, sequence_len) with 1s everywhere and 0 for values which
        should be masked.
        """
        batch_size, seq_length = attention_mask.size()
        assert seq_length % block_size == 0, "Sequence length must be multiple of block size."
        assert n_global_tokens % block_size == 0, "Number of global tokens must be multiple of block size."

        n_blocks = seq_length // block_size
        n_global_blocks = n_global_tokens // block_size
        n_middle_blocks = n_blocks - n_global_blocks - 2
        blocked_mask = attention_mask.view(batch_size, seq_length // block_size, block_size)

        if n_middle_blocks >= 1:
            # Expand the block mask and shift just as for the shifted attention
            exp_blocked_mask = torch.cat([
                blocked_mask[:, n_global_blocks:n_global_blocks + n_middle_blocks, :],
                blocked_mask[:, n_global_blocks + 1:n_global_blocks + n_middle_blocks + 1, :],
                blocked_mask[:, n_global_blocks + 2:n_global_blocks + n_middle_blocks + 2, :]
            ], dim=2)  # [batch, n_middle_blocks, 3*block_size]

            inner_band_mask = torch.einsum("blq,blk->blqk",
                                           blocked_mask[:, n_global_blocks + 1:n_global_blocks + n_middle_blocks + 1, :],
                                           exp_blocked_mask
                                           )  # [batch, n_middle_block, block_size, 3*block_size]
            # unsqueeze for heads
            inner_band_mask.unsqueeze_(1)
        else:
            inner_band_mask = None

        # mask which is applied to the final output, i.e. masking tensors of size (batch, heads, seq_length, head_dim)
        from_mask = attention_mask.view(batch_size, 1, seq_length, 1)
        # mask which is applied to attentions, i.e. masking tensors of size (batch, heads, queries, seq_len)
        to_mask = attention_mask.view(batch_size, 1, 1, seq_length)

        if to_device:
            blocked_mask = blocked_mask.to(to_device)
            inner_band_mask = inner_band_mask.to(to_device) if inner_band_mask is not None else None
            from_mask = from_mask.to(to_device)
            to_mask = to_mask.to(to_device)

        return blocked_mask, inner_band_mask, from_mask, to_mask

    @staticmethod
    def torch_bmm_nd(inp_1, inp_2, ndim=None):
        """ Fast nd matrix multiplication """
        return torch.bmm(inp_1.reshape((-1,) + inp_1.shape[-2:]), inp_2.reshape((-1,) + inp_2.shape[-2:])).view(
            inp_1.shape[: ndim - 2] + (inp_1.shape[ndim - 2], inp_2.shape[ndim - 1])
        )

    @staticmethod
    def torch_bmm_nd_transpose(inp_1, inp_2, ndim=None):
        """ Fast nd matrix multiplication with transpose """
        return torch.bmm(
            inp_1.reshape((-1,) + inp_1.shape[-2:]), inp_2.reshape((-1,) + inp_2.shape[-2:]).transpose(1, 2)
        ).view(inp_1.shape[: ndim - 2] + (inp_1.shape[ndim - 2], inp_2.shape[ndim - 2]))


