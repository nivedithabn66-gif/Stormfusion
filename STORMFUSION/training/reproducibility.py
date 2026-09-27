"""STORMFUSION Training Reproducibility Utilities.

Provides global random seed configuration across Python, NumPy, PyTorch CPU, and PyTorch CUDA.

Note on Hardware Limitations:
Configuring random seeds guarantees reproducibility on identical hardware and software environments.
However, perfect bitwise determinism across fundamentally different GPU architectures, CUDA toolkit
versions, or CPU backends is not guaranteed by PyTorch.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Sets random seed across all execution frameworks.

    Args:
        seed: Integer seed value (default 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
