import jax
import jax.numpy as jnp
from jax import jit, grad
import equinox as eqx
import models.spline_autograd_func as spline_autograd_func
import time


class LinearSpline(eqx.Module):
    """
    Class for LinearSpline activation functions

    Args:
        num_knots (int): number of knots of the spline
        num_activations (int) : number of activation functions
        x_min (float): position of left-most knot
        x_max (float): position of right-most knot
        slope_min (float or None): minimum slope of the activation
        slope_max (float or None): maximum slope of the activation
        antisymmetric (bool): Constrain the potential to be symmetric <=> activation antisymmetric
    """
    num_activations: int 
    num_knots: int 
    x_min: jnp.ndarray 
    x_max: jnp.ndarray
    init: str 
    slope_min: float 
    slope_max: float 
    antisymmetric: bool
    clamp: bool 
    step_size: jnp.ndarray
    no_constraints: bool
    integrated_coeff: jnp.ndarray
    coefficients: jnp.ndarray
    projected_coefficients_cached: jnp.ndarray
    zero_knot_indexes: jnp.ndarray
    zero_knot_indexes_integrated: jnp.ndarray

    def __init__(self, num_activations, num_knots, x_min, x_max, init, slope_max=None, slope_min=None, antisymmetric=False, clamp=True, **kwargs):
        
        # super().__init__()
        self.num_knots = int(num_knots)
        self.num_activations = int(num_activations)
        self.init = init
        self.x_min = jnp.array([x_min])
        self.x_max = jnp.array([x_max])
        self.slope_min = slope_min
        self.slope_max = slope_max

        self.step_size = (self.x_max - self.x_min) / (self.num_knots - 1)
        self.antisymmetric = antisymmetric
        self.clamp = clamp
        self.no_constraints = (slope_max is None and slope_min is None and (not antisymmetric) and not clamp)
        self.integrated_coeff = None
        
        # parameters
        coefficients = self.initialize_coeffs()  # spline coefficients
        self.coefficients = coefficients

        self.projected_coefficients_cached = None
        self.zero_knot_indexes_integrated = None
        self.init_zero_knot_indexes()


    def init_zero_knot_indexes(self):
        """ Initialize indexes of zero knots of each activation.
        """
        # self.zero_knot_indexes[i] gives index of knot 0 for filter/neuron_i.
        # size: (num_activations,)
        activation_arange = jnp.arange(0, self.num_activations)
        self.zero_knot_indexes = (activation_arange * self.num_knots)

    def initialize_coeffs(self):
        """The coefficients are initialized with the value of the activation
        # at each knot (c[k] = f[k], since B1 splines are interpolators)."""

        init = self.init
        grid_tensor = jnp.tile(jnp.linspace(self.x_min.item(), self.x_max.item(), self.num_knots), (self.num_activations, 1))

        if isinstance(init, float):
            coefficients = jnp.ones_like(grid_tensor) * init
        elif init == 'identity':
            coefficients = grid_tensor
        elif init == 'zero':
            coefficients = grid_tensor*0
        else:
            raise ValueError('init should be in [identity, zero].')
        
        return coefficients
    
    @property
    def projected_coefficients(self):
        """ B-spline coefficients projected to meet the constraint. """
        if self.projected_coefficients_cached is not None:
            return self.projected_coefficients_cached
        else:
            return self.clipped_coefficients()

    def cached_projected_coefficients(self):
        """ B-spline coefficients projected to meet the constraint. """
        if self.projected_coefficients_cached is None:
            self.projected_coefficients_cached = self.clipped_coefficients()
    
    @property
    def slopes(self):
        """ Get the slopes of the activations """
        coeff = self.projected_coefficients
        slopes = (coeff[:, 1:] - coeff[:, :-1]) / self.step_size
        return slopes
    
    @property
    def device(self):
        return self.coefficients.device
    
    def hyper_param_to_device(self):
        device = self.device
        # self.x_min, self.x_max, self.step_size, self.zero_knot_indexes = self.x_min.to_device(device), self.x_max.to_device(device), self.step_size.to_device(device), self.zero_knot_indexes.to_device(device)

        
    def forward(self, x):
        """
        Args:
            input (jax.numpy.ndarray):
                2D or 4D, depending on weather the layer is
                convolutional ('conv') or fully-connected ('fc')

        Returns:
            output (jax.numpy.ndarray)
        """
        self.hyper_param_to_device()

        in_shape = x.shape
        in_channels = in_shape[1]

        if in_channels % self.num_activations != 0:
            raise ValueError('Number of input channels must be divisible by number of activations.')
        
        x = x.reshape(x.shape[0], self.num_activations, in_channels // self.num_activations, *x.shape[2:])

        x = spline_autograd_func.linear_spline_func(x, self.projected_coefficients, self.x_min, self.x_max, self.num_knots, self.zero_knot_indexes)[0]

        x = x.reshape(in_shape)

        return x
    
    def __call__(self, x):
        return self.forward(x)
    

    def derivative(self, x):
        """
        Args:
            input (jax.numpy.ndarray):
                2D or 4D, depending on weather the layer is
                convolutional ('conv') or fully-connected ('fc')

        Returns:
            output (jax.numpy.ndarray)
        """

        self.hyper_param_to_device()

        in_shape = x.shape
        in_channels = in_shape[1]

        if in_channels % self.num_activations != 0:
            raise ValueError('Number of input channels must be divisible by number of activations.')
        
        x = x.reshape(-1, self.num_activations, in_channels // self.num_activations, *x.shape[2:])

        coeff = self.projected_coefficients

        x = spline_autograd_func.linear_spline_derivative_func(x, coeff, self.x_min, self.x_max, self.num_knots, self.zero_knot_indexes)[0]

        x = x.reshape(in_shape)

        return x 
    

    def update_integrated_coeff(self):
        print("**** Updating integrated spline coefficients ****")
        coeff = self.projected_coefficients

        # extrapolate assuming zero slopes at both ends, i.e. linear interpolation for the integrated function
        coeff_int = jnp.concatenate((coeff[:, 0:1], coeff, coeff[:, -1:]), axis=1)

        # integrate to obtain
        # the coefficents of the corresponding quadratic BSpline expansion
        object.__setattr__(self,'integrated_coeff', jnp.cumsum(coeff_int, axis=1)*self.step_size ) 
        # self.integrated_coeff = jnp.cumsum(coeff_int, axis=1)*self.step_size

        # remove value at 0 and reshape
        # this is arbitray, as integration is up to a constant
        object.__setattr__(self,'integrated_coeff',(self.integrated_coeff - self.integrated_coeff[:, (self.num_knots + 2) // 2].reshape(-1, 1)).reshape(-1))
        # self.integrated_coeff = (self.integrated_coeff - self.integrated_coeff[:, (self.num_knots + 2) // 2].reshape(-1, 1)).reshape(-1)

        # store once for all knots indexes
        # not the same as for the linear-spline as we have 2 more "virtual" knots now
        object.__setattr__(self, 'zero_knot_indexes_integrated', jnp.arange(0, self.num_activations) * (self.num_knots + 2))
        # self.zero_knot_indexes_integrated = jnp.arange(0, self.num_activations) * (self.num_knots + 2)


    def integrate(self, x):
        in_shape = x.shape

        in_channels = in_shape[1]

        if in_channels % self.num_activations != 0:
            raise ValueError('Number of input channels must be divisible by number of activations.')
        
        if self.integrated_coeff is None:
            self.update_integrated_coeff()

        if x.device != self.integrated_coeff.device:
            self.integrated_coeff = self.integrated_coeff.to_device(x.device)
            self.zero_knot_indexes_integrated = self.zero_knot_indexes_integrated.to_device(x.device)

        x = x.reshape(-1, self.num_activations, in_channels // self.num_activations, *x.shape[2:])

        x = spline_autograd_func.quadratic_spline_func(x - self.step_size, self.integrated_coeff, self.x_min - self.step_size, self.x_max + self.step_size, self.num_knots + 2, self.zero_knot_indexes_integrated)[0]
        
        x = x.reshape(in_shape)

        return (x)
    
    def extra_repr(self):
        """ repr for print(model) """
        s = ('num_activations={num_activations}, '
             'init={init}, num_knots={num_knots}, range=[{x_min[0]:.3f}, {x_max[0]:.3f}], '
             'slope_max={slope_max}, '
             'slope_min={slope_min}.'
             )
        return s.format(**self.__dict__)
    
    def clipped_coefficients(self):        
        """Simple projection of the spline coefficients to enforce the constraints, for e.g. bounded slope"""

        device = self.device

        if self.no_constraints:
            return(self.coefficients)
            
        cs = self.coefficients

        new_slopes = (cs[:, 1:] - cs[:, :-1]) / self.step_size

        if self.slope_min is not None or self.slope_max is not None:
            new_slopes = jnp.clip(new_slopes, self.slope_min, self.slope_max)

        # clamp extension
        if self.clamp:
            new_slopes = new_slopes.at[:, 0].set(0)
            new_slopes = new_slopes.at[:, -1].set(0)

        new_cs = jnp.zeros(self.coefficients.shape, device=device, dtype=cs.dtype)

        new_cs = new_cs.at[:,1:].set(jnp.cumsum(new_slopes, axis=1) * self.step_size)

        
        # preserve the mean, unless antisymmetric
        if not self.antisymmetric:
            new_cs = new_cs + new_cs.mean(axis=1).reshape(-1, 1)

        # antisymmetry
        if self.antisymmetric:
            inv_idx = jnp.arange(new_cs.shape[1] - 1, -1, -1).to_device(new_cs.device)
            inv_tensor = new_cs[:, inv_idx]
            new_cs = 0.5 * (new_cs - inv_tensor)
        
        return new_cs
    

    # tranform the splines into clip functions
    def get_clip_equivalent(self):
        self.hyper_param_to_device()
        coeff_proj = self.projected_coefficients.clone().to_device(self.device)
        slopes = (coeff_proj[:,1:] - coeff_proj[:,:-1])
        slopes_change = slopes[:,1:] - slopes[:,:-1]

        i1 = jnp.argmax(slopes_change, axis=1)
        i2 = jnp.argmin(slopes_change, axis=1)

        i0 = jnp.arange(0, coeff_proj.shape[0]).to_device(coeff_proj.device)

        grid_tensor = jnp.tile(jnp.linspace(self.x_min.item(), self.x_max.item(), self.num_knots, device= self.device), (self.num_activations, 1))
        x1 = grid_tensor[i0, i1 + 1].reshape(1, -1, 1, 1)
        y1 = coeff_proj[i0, i1 + 1].reshape(1, -1, 1, 1)

        x2 = grid_tensor[i0, i2 + 1].reshape(1, -1, 1, 1)
        y2 = coeff_proj[i0, i2 + 1].reshape(1, -1, 1, 1)

        slopes = ((y2 - y1) / (x2 - x1)).reshape(1, -1, 1, 1)

        cl = clip_activation(x1, x2, y1, slopes)

        return(cl)

class clip_activation(eqx.Module):
    x1: jnp.ndarray
    x2: jnp.ndarray
    slopes: jnp.ndarray
    y1: jnp.ndarray

    def __init__(self, x1, x2, y1, slopes):
        self.x1 = x1
        self.x2 = x2
        self.slopes = slopes
        self.y1 = y1

    def forward(self, x):
        return (self.slopes * (jax.nn.relu(x - self.x1) - jax.nn.relu(x - self.x2)) + self.y1)
    
    def __call__(self, x):
        return self.forward(x)
    
    def integrate(self, x):
        return(self.slopes/2 * ((jax.nn.relu(x - self.x1)**2 - jax.nn.relu(x - self.x2)**2) + self.y1 * x))
    
    @property
    def slope_max(self):
        slope_max = jnp.max(self.slopes, axis=1)
        return (slope_max)