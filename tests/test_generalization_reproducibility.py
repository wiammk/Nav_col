import random

import numpy as np
import torch

from scripts.evaluation.evaluate_generalization import seed_everything


def test_seed_everything_repeats_python_numpy_and_torch_streams():
    seed_everything(123)
    first = (random.random(), float(np.random.random()), float(torch.rand(())))

    seed_everything(123)
    second = (random.random(), float(np.random.random()), float(torch.rand(())))

    assert first == second
