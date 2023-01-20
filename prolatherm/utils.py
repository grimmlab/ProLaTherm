import torch
import random
import numpy as np
import pathlib
from itertools import groupby


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


def read_fasta(filepath: pathlib.Path) -> (np.array, np.array):
    """
    Read protein FASTA files (.fasta, .txt), i.e. files that contain pais of protein ID and amino aid sequence:

    >protein1|some identifier|another identifier|
    MGKRVVIALGGNALQQRGQKGSYEEMMDNVRKTARQIAEIIARGYEVVIT
    MSTESEIAVRIRGIYSTALTKLLMDRGFKIVQPSDVIAERFGIEKSYEDF
    DVDIYDKNHGVTIVGTKVEAVKKVFEEEFIDVFFRKLPYKLHGIYKGLVV
    KRDDRFVYVDIGNVIGTVLIEELPDAAEGDEVVVQVKKHNVLPHLSTLIT
    >protein2|some identifier|another identifier|
    GLEDVYIDQTNICYIDGKEGKLYYRGYSVEELAELSTFEEVVYLLEIIAE

    Will return sequences and corresponding IDs (= the complete first line without '>')

    :param filepath: full path of FASTA file

    :return: sequences and corresponding IDs in two separate numpy arrays
    """
    with open(filepath, "r") as f:
        sequences = []
        protein_ids = []
        groups = (x[1] for x in groupby(f, lambda line: line[0] == ">"))
        for header in groups:
            pid = header.__next__().replace('>', '').strip()
            seq = "".join(s.strip() for s in groups.__next__())  # join all sequence lines to one.
            protein_ids.append(pid)
            sequences.append(seq)
    return np.array(sequences), np.array(protein_ids)
