import torch
import random
import numpy as np


def set_all_seeds(seed: int = 42):
    """
    Set all seeds of libs with a specific function for reproducibility of results

    :param seed: seed to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

