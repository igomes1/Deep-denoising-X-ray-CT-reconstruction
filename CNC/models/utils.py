import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import os
import glob
import json
from pathlib import Path

def accelerated_gd_batch(x_noisy, model, sigma=None, ada_restart=False, stop_condition=None, lmbd=1, grad_op=None, **kwargs):
    """compute the proximal operator using the FISTA accelerated rule"""

    max_iter = kwargs.get('max_iter', 500)
    tol = kwargs.get('tol', 1e-4)

    # initial value: noisy image
    x = torch.clone(x_noisy)
    z = torch.clone(x_noisy)
    t = torch.ones(x.shape[0], device=x.device).view(-1,1,1,1)