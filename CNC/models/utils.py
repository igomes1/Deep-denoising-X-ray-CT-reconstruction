import jax
import jax.numpy as jnp
import equinox as eqx
import math
import os
import glob

def accelerated_gd_batch(x_noisy, model, sigma=None, ada_restart=False, stop_condition=None, lmbd=1, grad_op=None, **kwargs):
    """compute the proximal operator using the FISTA accelerated rule"""

    max_iter = kwargs.get('max_iter', 500)
    tol = kwargs.get('tol', 1e-4)

    # initial value: noisy image
    x = x_noisy.clone()
    z = x_noisy.clone()
    t = jnp.ones(x.shape[0], device=x.device).reshape(-1,1,1,1)

    # cache values of scaling coeff for efficiency
    scaling = model.get_scaling(sigma=sigma)

    # the index of the images that have not converged yet
    idx = jnp.arange(0, x.shape[0], device=x.device)
    # relative change in the estimate
    res = jnp.ones(x.shape[0], device=x.device, dtype=x.dtype)

    # mean number of iterations over the batch
    i_mean = 0

    for i in range(max_iter):

        # model.scaling = scaling[idx]
        object.__setattr__(model, 'scaling', scaling[idx])
        x_old = x.clone()


        if grad_op is None:

            grad = model.grad_denoising(z[idx], x_noisy[idx], sigma=sigma[idx], cache_wx=False, lmbd=lmbd)
        else:
            grad = grad_op(z[idx])

        x = x.at[idx].set(z[idx] - grad)

        t_old = t.clone()
        t = 0.5 * (1 + jnp.sqrt(1 + 4*t**2))
        z = z.at[idx].set( x[idx] + (t_old[idx] - 1)/t[idx] * (x[idx] - x_old[idx]))

        if i > 0:
            res = res.at[idx].set(jnp.sqrt(jnp.sum((x[idx] - x_old[idx])**2, axis=(1,2,3))) / (jnp.sqrt(jnp.sum(x[idx]**2, axis=(1,2,3)))))



        if ada_restart:
            esti = jnp.sum(grad*(x[idx] - x_old[idx]), axis=(1,2,3))
            id_restart = jnp.where(esti > 0)[0]
            t = t.at[idx[id_restart]].set(1)
            z = z.at[idx[id_restart]].set(x[idx[id_restart]])

        condition = (res > tol)
        if stop_condition is None:
            idx = jnp.where(condition)[0]

        i_mean += jnp.sum(condition).item() / x.shape[0]


        if stop_condition is None:
            if jnp.max(res) < tol:
                break
        else:

            sct = stop_condition(x, i)

            if sct:
                break


    model.clear_scaling()
    return(x, i, i_mean)

def accelerated_gd_single(x_noisy, model, sigma=None, ada_restart=False, stop_condition=None, lmbd=1, grad_op=None, t_init=1, **kwargs):
    """compute the proximal operator using the FISTA accelerated rule"""

    max_iter = kwargs.get('max_iter', 500)
    tol = kwargs.get('tol', 1e-4)

    # initial value: noisy image
    x = x_noisy.clone()
    z = x_noisy.clone()
    t = t_init

    model.clear_scaling()
    model.scaling = model.get_scaling(sigma=sigma)
    object.__setattr__(model, 'scaling', model.get_scaling(sigma=sigma))
    # the index of the images that have not converged yet
    # relative change in the estimate
    res = 100000


    for i in range(max_iter):


        x_old = x.clone()

        if grad_op is None:
            grad = model.grad_denoising(z, x_noisy, sigma=sigma, cache_wx=False, lmbd=lmbd)
        else:
            grad = grad_op(z)

        x = z - grad


        t_old = t
        t = 0.5 * (1 + math.sqrt(1 + 4*t**2))
        z = x + (t_old - 1)/t * (x - x_old)

        if i > 0:
            res = (jnp.linalg.norm(x - x_old) / (jnp.linalg.norm(x))).item()

        


        if ada_restart:
            esti = jnp.sum(grad*(x[idx] - x_old[idx]), axis=(1,2,3))
            id_restart = jnp.where(esti > 0)[0]
            if len(id_restart) > 0:
                print(i, " restart", len(id_restart))
            t= t.at[idx[id_restart]].set(1)
            z= z.at[idx[id_restart]].set(x[idx[id_restart]])


        condition = (res > tol)
        if stop_condition is None:
            idx = jnp.where(condition)[0]


        if stop_condition is None:
            if jnp.max(res) < tol:
                break
        else:

            sct = stop_condition(x, i)

            if sct:
                break
    

    model.clear_scaling()
    return(x, i, t)
            
import sys
sys.path.append('../')
from models.wc_conv_net import WCvxConvNet
from pathlib import Path
import json

def load_model(name, device='cpu', epoch=None):
    current_directory = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute()
    directory = f'{current_directory}/trained_models/{name}/'
    directory_checkpoints = f'{directory}checkpoints/'

    rep = {"[": "[[]", "]": "[]]"}

    name_glob = name.replace("[","tr1u").replace("]","tr2u").replace("tr1u","[[]").replace("tr2u","[]]")
    
    print(f'{current_directory}/trained_models/{name_glob}/checkpoints/*.eqx')
    if epoch is None:
        files = glob.glob(f'{current_directory}/trained_models/{name_glob}/checkpoints/*.eqx', recursive=False)
        epochs = map(lambda x: int(x.split("/")[-1].split('.eqx')[0].split('_')[1]), files)
        epoch = max(epochs)

    checkpoint_path = f'{directory_checkpoints}checkpoint_{epoch}.eqx'
    # config file
    config = json.load(open(f'{directory}config.json'.replace("[[]","[").replace("[]]","]")))

    # build model

    model, _ = build_model(config)
    # model.to_device(device)
    model = eqx.tree_deserialise_leaves(checkpoint_path, model)
    model.conv_layer.spectral_norm()
    model.eval()############### A DMD ET JE PENSE à FAIRE car y a pas de method .eval() dans model

    return(model)


def build_model(config):
    # ensure consistency of the config file, e.g. number of channels, ranges + enforce constraints

    # 1- Activation function (learnable spline)
    param_spline_activation = config['spline_activation']
    # non expansive increasing splines
    param_spline_activation["slope_min"] = 0
    param_spline_activation["slope_max"] = 1
    # antisymmetric splines
    param_spline_activation["antisymmetric"] = True
    # shared spline
    param_spline_activation["num_activations"] = 1

    # 2- Multi convolution layer
    param_multi_conv = config['multi_convolution']
    if len(param_multi_conv['num_channels']) != (len(param_multi_conv['size_kernels']) + 1):
        raise ValueError("Number of channels specified is not compliant with number of kernel sizes")
    
    param_spline_scaling = config['spline_scaling']
    param_spline_scaling["clamp"] = False
    param_spline_scaling["x_min"] = config['noise_range'][0]
    param_spline_scaling["x_max"] = config['noise_range'][1]
    param_spline_scaling["num_activations"] = config['multi_convolution']['num_channels'][-1]


    ###########    QUESTION add a specific seed ?
    model = WCvxConvNet(param_multi_conv=param_multi_conv, param_spline_activation=param_spline_activation, param_spline_scaling=param_spline_scaling, rho_wcvx=config["rho_wcvx"])

    return(model, config)