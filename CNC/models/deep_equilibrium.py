import jax
import jax.numpy as jnp
import equinox as eqx
import sys
from functools import partial

sys.path.append("..")
from models.utils import accelerated_gd_batch

class DEQFixedPoint(eqx.Module):
    model: eqx.Module
    solver: callable
    params_fw: dict
    params_bw: dict
    f: callable 
    fjvp: callable 
    forward_niter_max: int 
    forward_niter_mean: float 
    backward_niter: int 
    

    def __init__(self, model, params_fw, params_bw):
        self.model = model
        self.solver = anderson
        self.params_fw = params_fw
        self.params_bw = params_bw
        self.f = None
        self.fjvp = None

    def __call__(self, x_noisy, sigma=None, **kwargs):
        # update spectral norm of the convolutional layer
        if self.model.training:
            self.model.conv_layer.spectral_norm()

        # fixed point iteration
        def f(x, x_noisy):
                return(x - self.model.grad_denoising(x, x_noisy, sigma=sigma))
        
        # jacobian vector product of the fixed point iteration
        def fjvp(x, y):
            return(y - self.model.hvp_denoising(x, y, sigma=sigma))
        
        object.__setattr__(self, 'f', f)
        object.__setattr__(self, 'fjvp', fjvp)
        # self.f = f
        # self.fjvp = fjvp

        # compute the fixed point
        def deq_fwd(x_noisy, sigma):
            z, forward_niter_max, forward_niter_mean = accelerated_gd_batch(x_noisy, self.model, sigma=sigma, ada_restart=True, **self.params_fw)
            object.__setattr__(self, 'forward_niter_max', forward_niter_max)
            object.__setattr__(self, 'forward_niter_mean', forward_niter_mean)
            z = f(z, x_noisy)
            z0 = jax.lax.stop_gradient(z.clone())
            return z, z0


        def deq_bwd(z0, grad):
            z0 = jax.lax.stop_gradient(z0) # jsp trop où le mettre
            g, backward_niter = self.solver(lambda y : self.fjvp(z0, y) + grad,
                                            grad, **self.params_bw)
            object.__setattr__(self, 'backward_niter', backward_niter)
            return (g, None)
        
        
        if self.model.training:
            deq_call = jax.custom_vjp(lambda x_noisy, sigma: deq_fwd(x_noisy, sigma)[0])
            deq_call.defvjp(deq_fwd, deq_bwd)
        else:
            deq_call = lambda x_noisy, sigma: deq_fwd(x_noisy, sigma)[0]

        z = deq_call(x_noisy, sigma)

        return z


def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-5, beta = 1.0):
    """ Anderson acceleration for fixed point iteration. """
    bsz, d, H, W = x0.shape
    X = jnp.zeros((bsz, m, d*H*W), dtype=x0.dtype, device=x0.device)
    F = jnp.zeros((bsz, m, d*H*W), dtype=x0.dtype, device=x0.device)
    X = X.at[:,0].set(x0.reshape(bsz, -1))
    F = F.at[:,0].set(f(x0).reshape(bsz, -1))
    X = X.at[:,1].set(F[:,0])
    F = F.at[:,1].set(f(F[:,0].reshape(x0.shape)).rehsape(bsz, -1))

    H = jnp.zeros((bsz, m+1, m+1), dtype=x0.dtype, device=x0.device)
    H = H.at[:,0,1:].set(1)
    H = H.at[:,1:,0].set(1)
    y = jnp.zeros((bsz, m+1, 1), dtype=x0.dtype, device=x0.device)
    y = y.at[:,0].set(1)

    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        H = H.at[:,1:n+1,1:n+1].set(jax.lax.batch_matmul(G,G.swapaxes(1,2)) + lam*jnp.eye(n, dtype=x0.dtype,device=x0.device)[None])

        alpha = jnp.linalg.solve(H[:, :n + 1, :n + 1], y[:, :n + 1])[:,1:n+1,0]
        X = X.at[:,k%m].set(beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0])
        F = F.at[:,k%m].set(f(X[:,k%m].reshape(x0.shape)).reshape(bsz, -1))
        res.append(jnp.linalg.norm(F[:,k%m] - X[:,k%m]).item()/(1e-5 + jnp.linalg.norm(F[:,k%m]).item()))
        if (res[-1] < tol):
            break

    return X[:,k%m].reshape(x0.shape), k