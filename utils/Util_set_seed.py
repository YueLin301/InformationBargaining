import numpy as np
import torch, os
import random


def all_seed(env, seed=1):
    if env:
        env.seed(seed)  # env config
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # config for CPU
    torch.cuda.manual_seed(seed)  # config for GPU
    os.environ["PYTHONHASHSEED"] = str(seed)  # config for python scripts
    # config for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
