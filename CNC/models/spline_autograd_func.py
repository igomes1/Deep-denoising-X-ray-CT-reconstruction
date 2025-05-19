import jax
import jax.numpy as jnp

@jax.custom_vjp
def linear_spline_func(x, coefficients, x_min, x_max, num_knots, zero_knot_indexes):
    """
    Autograd function to only backpropagate through the B-splines that were
    used to calculate output = activation(input), for each element of the
    input.
    """

    # The value of the spline at any x is a combination 
    # of at most two coefficients
    step_size = (x_max - x_min) / (num_knots - 1)
    x_clamped = x.clip(min=x_min.item(), max=x_max.item() - step_size.item())

    floored_x = jnp.floor((x_clamped - x_min) / step_size)  #left coefficient

    fracs = (x - x_min) / step_size - floored_x  # distance to left coefficient

    # This gives the indexes (in coefficients_vect) of the left
    # coefficients
    indexes = (zero_knot_indexes.reshape((1, -1, 1, 1, 1)) + floored_x).astype(jnp.int32)

    coefficients_vect = coefficients.reshape(-1)
    # Only two B-spline basis functions are required to compute the output
    # (through linear interpolation) for each input in the B-spline range.

    activation_output = coefficients_vect[indexes + 1] * fracs + coefficients_vect[indexes] * (1 - fracs)
    
    # save_for_backward
    ctx = (fracs, coefficients, indexes, step_size)

    return activation_output, ctx

def linear_spline_func_fwd(x, coefficients, x_min, x_max, num_knots, zero_knot_indexes):
    activation_output, ctx = linear_spline_func(x, coefficients, x_min, x_max, num_knots, zero_knot_indexes)
    return activation_output, ctx


def linear_spline_func_bwd(ctx, grad_out):
    fracs, coefficients, indexes, step_size = ctx

    coefficients_vect = coefficients.reshape(-1)

    grad_x = (coefficients_vect[indexes + 1] - coefficients_vect[indexes]) / step_size * grad_out

    # Next, add the gradients with respect to each coefficient, such that,
    # for each data point, only the gradients wrt to the two closest
    # coefficients are added (since only these can be nonzero).

    grad_coefficients_vect = jnp.zeros_like(coefficients_vect, dtype=coefficients_vect.dtype)

    # right coefficients gradients
    grad_coefficients_vect = grad_coefficients_vect.at[indexes.reshape(-1) + 1].add((fracs * grad_out).reshape(-1))
    
    # left coefficients gradients
    grad_coefficients_vect = grad_coefficients_vect.at[indexes.reshape(-1)].add(((1 - fracs) * grad_out).reshape(-1))

    grad_coefficients = grad_coefficients_vect.reshape(coefficients.shape)

    return grad_x, grad_coefficients, None, None, None, None

# link forward and backward operator
linear_spline_func.defvjp(linear_spline_func_fwd, linear_spline_func_bwd)



@jax.custom_vjp
def linear_spline_derivative_func(x, coefficients, x_min, x_max, num_knots, zero_knot_indexes):
    """
    Autograd function to only backpropagate through the B-splines that were
    used to calculate output = activation(input), for each element of the
    input.
    """
    # The value of the spline at any x is a combination 
    # of at most two coefficients
    step_size = (x_max - x_min) / (num_knots - 1)
    x_clamped = x.clip(min=x_min.item(), max=x_max.item() - step_size.item())

    floored_x = jnp.floor((x_clamped - x_min) / step_size)  #left coefficient

    fracs = (x - x_min) / step_size - floored_x  # distance to left coefficient

    # This gives the indexes (in coefficients_vect) of the left
    # coefficients
    indexes = (zero_knot_indexes.reshape(1, -1, 1, 1, 1) + floored_x).astype(jnp.int32)


    coefficients_vect = coefficients.reshape(-1)
    # Only two B-spline basis functions are required to compute the output
    # (through linear interpolation) for each input in the B-spline range.
    activation_output = (coefficients_vect[indexes + 1] - coefficients_vect[indexes]) / step_size

    # save_for_backward
    ctx = (fracs, coefficients, indexes, step_size)
    return activation_output, ctx

def linear_spline_derivative_func_fwd(x, coefficients, x_min, x_max, num_knots, zero_knot_indexes):
    activation_output, ctx = linear_spline_derivative_func(x, coefficients, x_min, x_max, num_knots, zero_knot_indexes)
    return activation_output, ctx

def linear_spline_derivative_func_bwd(ctx, grad_out):
    fracs, coefficients, indexes, step_size = ctx
    grad_x = 0 * grad_out

    # Next, add the gradients with respect to each coefficient, such that,
    # for each data point, only the gradients wrt to the two closest
    # coefficients are added (since only these can be nonzero).

    grad_coefficients_vect = jnp.zeros_like(coefficients.reshape(-1))
    # right coefficients gradients
    grad_coefficients_vect = grad_coefficients_vect.at[indexes.reshape(-1) + 1].add(jnp.ones_like(fracs).reshape(-1) / step_size)

    # left coefficients gradients
    grad_coefficients_vect = grad_coefficients_vect.at[indexes.reshape(-1)].add(-jnp.ones_like(fracs).reshape(-1) / step_size)

    return grad_x, grad_coefficients_vect, None, None, None, None

# link forward and backward operator
linear_spline_derivative_func.defvjp(linear_spline_derivative_func_fwd, linear_spline_derivative_func_bwd)

@jax.custom_vjp
def quadratic_spline_func(x, coefficients, x_min, x_max, num_knots, zero_knot_indexes):
    """
    Autograd function to only backpropagate through the B-splines that were
    used to calculate output = activation(input), for each element of the
    input.
    """

    step_size = (x_max - x_min) / (num_knots - 1)
    x_clamped = x.clip(min=x_min.item(), max=x_max.item() - 2*step_size.item())

    floored_x = jnp.floor((x_clamped - x_min) / step_size)  #left 

    # This gives the indexes (in coefficients_vect) of the left
    # coefficients
    indexes = (zero_knot_indexes.reshape(1, -1, 1, 1, 1) + floored_x).astype(jnp.int32)

    # B-Splines evaluation
    shift1 = (x - x_min) / step_size - floored_x

    frac1 = ((shift1 - 1)**2)/2
    frac2 = (-2*(shift1)**2 + 2*shift1 + 1)/2 
    frac3 = (shift1)**2/2

    coefficients_vect = coefficients.reshape(-1)

    activation_output = coefficients_vect[indexes + 2] * frac3 + coefficients_vect[indexes + 1] * frac2 + coefficients_vect[indexes] * frac1

    grad_x = coefficients_vect[indexes + 2] * (shift1) + coefficients_vect[indexes + 1] * (1 - 2*shift1) + coefficients_vect[indexes] * ((shift1 - 1))

    grad_x = grad_x / step_size

    # save_for_backward
    ctx = (grad_x, frac1, frac2, frac3, coefficients, indexes, step_size)

    return activation_output, ctx


def quadratic_spline_func_fwd(x, coefficients, x_min, x_max, num_knots, zero_knot_indexes):
    activation_output, ctx = quadratic_spline_func(x, coefficients, x_min, x_max, num_knots, zero_knot_indexes)
    return activation_output, ctx 


def quadratic_spline_func_bwd(ctx, grad_out):

    grad_x, frac1, frac2, frac3, coefficients, indexes, grid = ctx 

    coefficients_vect = coefficients.reshape(-1)

    grad_x = grad_x * grad_out

     # Next, add the gradients with respect to each coefficient, such that,
    # for each data point, only the gradients wrt to the two closest
    # coefficients are added (since only these can be nonzero).

    grad_coefficients_vect = jnp.zeros_like(coefficients_vect)
    # coefficients gradients
    grad_coefficients_vect = grad_coefficients_vect.at[indexes.reshape(-1) + 2].add((frac3 * grad_out).reshape(-1))

    grad_coefficients_vect = grad_coefficients_vect.at[indexes.reshape(-1) + 1].add((frac2 * grad_out).reshape(-1))

    grad_coefficients_vect = grad_coefficients_vect.at[indexes.reshape(-1)].add((frac1 * grad_out).reshape(-1))

    grad_coefficients = grad_coefficients_vect.reshape(coefficients.shape)

    return grad_x, grad_coefficients, None, None, None, None


quadratic_spline_func.defvjp(quadratic_spline_func_fwd, quadratic_spline_func_bwd)
