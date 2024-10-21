import random
import numpy as np
import torch

RS_RANDOM = 0
RS_NUMPY = 0
RS_TORCH = 0

def set_random_seeds(
        numpy_seed=RS_RANDOM,
        random_seed=RS_NUMPY,
        torch_seed=RS_TORCH,
        ):
    random.seed(random_seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)