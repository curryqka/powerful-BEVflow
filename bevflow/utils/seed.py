"""固定随机种子：python，numpy，torch，torch.cuda。"""
import os
import random

import numpy as np
import torch


def fix_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
