import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Sequence
import numpy as np


class MultiConv2d(eqx.Module):
    conv_layers: list
    size_kernels: Sequence[int]
    num_channels: Sequence[int]
    sn_size: int
    zero_mean: bool
    L: jnp.ndarray 
    dirac: jnp.ndarray 
    padding_total: int
    

    def __init__(self, num_channels=[1, 64], size_kernels=[3], zero_mean=True, sn_size=256, seed = 0):
         
        # parameters and options
        self.size_kernels = size_kernels
        self.num_channels = num_channels
        self.sn_size = sn_size
        self.zero_mean = zero_mean

        
        # list of convolutionnal layers
        self.conv_layers = []
        # keys to provide randomness for parameter initialisation
        keys = jax.random.split(jax.random.PRNGKey(seed), len(num_channels) - 1)
        for j in range(len(num_channels) - 1):
            layer = eqx.nn.Conv2d(in_channels=num_channels[j], out_channels=num_channels[j+1], kernel_size= size_kernels[j], padding= size_kernels[j]//2, stride=1, use_bias= False, key= keys[j])
           
           # enforce zero mean filter for first conv
            if zero_mean and j == 0:
                layer = eqx.tree_at(lambda c: c.weight,layer,ZeroMean()(layer.weight))

            self.conv_layers.append(layer)

        
        # cache the estimation of the spectral norm
        self.L = jnp.array(1.0)
        # cache dirac impulse used to estimate the spectral norm
        self.padding_total = sum([k // 2 for k in size_kernels])
        self.dirac = jnp.zeros((1, 1, 4 * self.padding_total + 1, 4 * self.padding_total + 1))
        self.dirac = self.dirac.at[0, 0, 2 * self.padding_total, 2 * self.padding_total].set(1)


    def forward(self, x):
        return(self.convolution(x))
    
    def __call__(self, x):
        return self.forward(x)

    def convolution(self, x):
        # normalized convolution, so that the spectral norm of the convolutional kernel is 1
        # nb the spectral norm of the convolution has to be upated before
        x = x / jnp.sqrt(self.L)

        for conv in self.conv_layers:
            weight = conv.weight
            x =  jax.lax.conv_general_dilated(x, weight, window_strides=conv.stride, padding=conv.padding, lhs_dilation=(1,1), rhs_dilation=conv.dilation, dimension_numbers=('NCHW', 'OIHW', 'NCHW'))

        return x
    
    def transpose(self, x):
        # normalized transpose convolution, so that the spectral norm of the convolutional kernel is 1
        # nb the spectral norm of the convolution has to be upated before
        x = x / jnp.sqrt(self.L)

        for conv in reversed(self.conv_layers):

            weight = conv.weight
            lhs_spec = ("NCHW")  # input
            rhs_spec = ("OIHW")  # weights
            out_spec = ("NCHW")

            # jax.lax.conv_transpose attends les strides par dimension spatiale uniquement
            x = jax.lax.conv_transpose(lhs=x, rhs=weight, strides=conv.stride, padding= conv.padding, dimension_numbers=(lhs_spec, rhs_spec, out_spec), transpose_kernel=True, precision=None )
        return x




    def spectral_norm(self, mode="Fourier", n_steps=1000, seed = None):
        """ Compute the spectral norm of the convolutional layer
                Args:
                    mode: "Fourier" or "power_method"
                        - "Fourier" computes the spectral norm by computing the DFT of the equivalent convolutional kernel. This is only an estimate (boundary effects are not taken into account) but it is differentiable and fast
                        - "power_method" computes the spectral norm by power iteration. This is more accurate and used before testing
                    n_steps: number of steps for the power method
                    seed:seed use for init power method
        """
        
        if mode == "Fourier":
            # temporary set L to 1 to get the spectral norm of the unnormalized filter
            object.__setattr__(self, 'L', jnp.array(1.0))
            # get the convolutional kernel corresponding to WtW
            kernel = self.get_kernel_WtW()
            # pad the kernel and compute its DFT. The spectral norm of WtW is the maximum of the absolute value of the DFT
            padding = (self.sn_size - 1) // 2 - self.padding_total
            padded = jnp.pad(kernel, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
            fft = jnp.fft.fft2(padded)
            object.__setattr__(self, 'L', jnp.max(jnp.abs(fft)))
            return self.L

        elif mode == "power_method":
            object.__setattr__(self, 'L', jnp.array(1.0))
            u = jax.random.normal(jax.random.PRNGKey(seed), (1, 1, self.sn_size, self.sn_size))
            u = jax.lax.stop_gradient(u)
            for _ in range(n_steps):
                u = self.transpose(self.convolution(u))
                u = u / jnp.linalg.norm(u)
              
            # The largest eigen value can now be estimated in a differentiable way
            sn = jnp.linalg.norm(self.transpose(self.convolution(u)))
            object.__setattr__(self, 'L', sn)
            return sn

    def check_tranpose(self, seed = 0):
            """
                Check that the convolutional layer is indeed the transpose of the convolutional layer
            """
            for i in range(1):
                x1 = jax.random.normal(jax.random.PRNGKey(seed), (1, 1, 40, 40))
                x2 = jax.random.normal(jax.random.PRNGKey(seed + 1), (1, self.num_channels[-1], 40, 40))
                ps_1 = jnp.sum(self.forward(x1) * x2)
                ps_2 = jnp.sum(self.transpose(x2) * x1)

                # x1_np = np.ones((1, 1, 40, 40), dtype=np.float32)  # Remplacer par des uns
                # x2_np = np.ones((1, self.num_channels[-1], 40, 40), dtype=np.float32)  # Remplacer par des uns
                # x1_np = jnp.array(x1_np)
                # x2_np = jnp.array(x2_np)
                ps_1 = jnp.sum(self.forward(x1) * x2)
                ps_2 = jnp.sum(self.transpose(x2) * x1)
                
                print(f"ps_1: {ps_1}")
                print(f"ps_2: {ps_2}")
                print(f"ratio: {ps_1 / ps_2}")

    def spectrum(self):
        kernel = self.get_kernel_WtW()
        padding = (self.sn_size - 1) // 2 - self.padding_total
        padded = jnp.pad(kernel, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
        return jnp.fft.fft2(padded)


    def get_filters(self):
        # we collapse the convolutions to get one kernel per channel
        # this done by computing the response of a dirac impulse
        kernel = self.convolution(self.dirac)[:,:,self.padding_total:3*self.padding_total+1, self.padding_total:3*self.padding_total+1]
        return kernel
    
    def get_kernel_WtW(self):
        return self.transpose(self.convolution(self.dirac))
    

class ZeroMean(eqx.Module):
    def __call__(self, weight: jnp.ndarray) -> jnp.ndarray:
        mean = jnp.mean(weight, axis=(1, 2, 3), keepdims=True)
        return weight - mean
