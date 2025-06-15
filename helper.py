# imports 
import jax 
import numpy as np
import jax.numpy as jnp
import scipy as sc
import matplotlib.pyplot as plt
import skimage as ski
from pathlib import Path
from tqdm import tqdm
import equinox as eqx

# global variable
img = ski.io.imread('data/img/concrete_surface.jpeg').astype(np.float32)
img = ski.color.rgb2gray(img)[:340, :340]
disk_img = np.zeros(img.shape)
disk_coord = ski.draw.disk((img.shape[0]//2,img.shape[1]//2), 150)
disk_img[disk_coord] = 1


##################################################################################################################################################
# Forward/backward Operations + filters
@jax.jit
def rotate_image(image, angle):
    """
    Rotate a 2D grayscale image by a specified angle.

    Parameters
    ----------
    image : jax.numpy.ndarray
        2D array representing the grayscale image to rotate.
    angle : float
        Rotation angle in degrees. Positive values correspond to counter-clockwise rotation.

    Returns
    -------
    rotated : jax.numpy.ndarray
        Rotated 2D image.
    """
    angle_rad = -angle * jnp.pi / 180  
    
    # Pixel coord.
    H, W = image.shape
    x, y = jnp.meshgrid(jnp.arange(W), jnp.arange(H))
    x = x - W // 2
    y = y - H // 2
    
    # Rotational matrix
    cos_theta = jnp.cos(angle_rad)
    sin_theta = jnp.sin(angle_rad)
    x_rot = cos_theta * x - sin_theta * y + W // 2
    y_rot = sin_theta * x + cos_theta * y + H // 2
    
    # Bilinear interpolation
    x0 = jnp.clip(jnp.floor(x_rot).astype(int), 0, W - 1)
    x1 = jnp.clip(x0 + 1, 0, W - 1)
    y0 = jnp.clip(jnp.floor(y_rot).astype(int), 0, H - 1)
    y1 = jnp.clip(y0 + 1, 0, H - 1)
    
    Ia = image[y0, x0]
    Ib = image[y0, x1]
    Ic = image[y1, x0]
    Id = image[y1, x1]
    
    wa = (x1 - x_rot) * (y1 - y_rot)
    wb = (x_rot - x0) * (y1 - y_rot)
    wc = (x1 - x_rot) * (y_rot - y0)
    wd = (x_rot - x0) * (y_rot - y0)
    
    rotated = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return rotated



@jax.jit
def radon_transform(img, angles): 
    """
    Compute the Radon transform (sinogram) of a 2D grayscale image for a set of projection angles.

    Parameters
    ----------
    img : jax.numpy.ndarray
        2D array representing the grayscale input image.
    angles : array-like
        Sequence of projection angles in degrees.

    Returns
    -------
    sinogram : jax.numpy.ndarray
        2D array of shape (L, len(angles)), where L is the diagonal length of the image.
        Each column contains the projection of the image at the corresponding angle.
    """
    dim_x, dim_y = img.shape
    diag = int(np.ceil(np.sqrt(dim_x**2 + dim_y**2)))  

    # pad img with zero due to rotation (max dimension that is projected = diag)
    padded_image = jnp.zeros((diag, diag))   
    
    # Center the img inside padded_img
    offset_x = (diag - dim_x) // 2
    offset_y = (diag - dim_y) // 2
    padded_image = padded_image.at[offset_x:offset_x + dim_x, offset_y:offset_y + dim_y].set(img)

   
    # projections
    sinogram = jnp.zeros((diag, len(angles)))  
    for i, angle in enumerate(angles):
        rotated = rotate_image(padded_image, angle)
        sinogram = sinogram.at[:,i].set(rotated.sum(axis=1))

    return sinogram




def ramp_fourier_filter(size, filter_name =None): 
    """Construct the ramp (Ram-Lak) filter in Fourrier domain. With optional modifications such as a cosine window.
    Taken from https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/transform/radon_transform.py#L127

    Parameters
    ----------
    size : int
        filter size. Must be even.
    filter_name : str or None, optional
        If None, the pure ramp filter is returned.
        If 'cosine', a cosine window is applied. 

    Returns
    -------
    fourier_filter: ndarray
        The computed Fourier filter.
    """

    n = np.concatenate(
        (
            np.arange(1, size / 2 + 1, 2, dtype=int),
            np.arange(size / 2 - 1, 0, -2, dtype=int),
        )
    )
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2

    fourier_filter = 2 * np.real(sc.fft.fft(f))  

    # if we want a modification of the filter ---> cosine filter 
    if filter_name == "cosine":
        freq = np.linspace(0, np.pi, size, endpoint=False)
        cosine_filter = sc.fft.fftshift(np.sin(freq)) 
        fourier_filter *= cosine_filter

    return fourier_filter[:, np.newaxis]
    



def fbp_par(sinogramm, angles, output_size, filter_name = None):
    """
    Perform Filtered Backprojection (FBP) reconstruction from a sinogram using parallel beam geometry.

    Parameters
    ----------
    sinogramm : ndarray of shape (N, M)
        Input sinogram where N is the number of detector pixels and M is the number of projection angles.
    angles : array-like of float
        List or array of projection angles in degrees.
    output_size : int
        Size (height and width) of the reconstructed square image.
    filter_name : str or None, optional
        Name of the filter applied in the Fourier domain.
        If None, a standard ramp filter is used.
        If 'cosine', a ramp filter with cosine window is applied.

    Returns
    -------
    recons_img : ndarray of shape (output_size, output_size)
        The reconstructed 2D image.
    """
        
    sino_shape = sinogramm.shape[0]

    #resize sinogramm next power of two for fourrier analyses
    projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * sino_shape))))
    pad_width = ((0, projection_size_padded - sino_shape), (0, 0))
    sinogramm_padded = np.pad(sinogramm, pad_width, mode='constant', constant_values=0)


    # apply Ramp filter 
    fourier_filter = ramp_fourier_filter(projection_size_padded, filter_name)
    projection = sc.fft.fft(sinogramm_padded, axis=0) * fourier_filter
    sinogramm_filtered = np.real(sc.fft.ifft(projection, axis=0)[:sino_shape, :])
    


    # back projection 
    recons_img = np.zeros((output_size, output_size))
    center = output_size//2
    xpr, ypr = np.mgrid[:output_size, :output_size] - center
    x = np.arange(sino_shape) - sino_shape // 2 
    for col, angle in zip(sinogramm_filtered.T, angles):
        t = xpr * np.cos(angle*np.pi/180) + ypr * np.sin(angle*np.pi/180) # t here is a matrix where the value of t[x,y] = t(x,y)

        # use linear interpolation
        recons_img += np.interp(t, xp=x, fp=col, left=0, right=0)

    return recons_img*np.pi/(2*len(angles))
    


##################################################################################################################################################
# Model-based algorithm 

class Model_Beam_par(eqx.Module):
    """
    Tomographic forward operator using the parallel-beam Radon transform.

    This model represents the forward operator R(x), which computes the
    Radon transform of an input image `x` over a set of projection angles.

    Attributes
    ----------
    angles : jnp.ndarray
        Array of projection angles in degrees, used to compute the sinogram.

    Methods
    -------
    __call__(x):
        Applies the Radon transform to the input image `x` using the specified angles.
    """
    angles: jnp.ndarray 

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        Rx = radon_transform(x, self.angles)
        return Rx
    


@jax.jit
def loss_L2(x, y, R):
    """
    Computes the squared Frobenius norm loss between the measured/target sinogram y
    and the projection R(x).

    Parameters
    ----------
    x : jnp.ndarray
        Reconstructed image (estimate).
    y : jnp.ndarray
        Measured sinogram (target).
    R : Callable
        Forward operator (e.g., an instance of Model_Beam_par) that maps x to its sinogram.

    Returns
    -------
    float
        Squared L2 loss (Frobenius norm) between y and R(x).
    """ 
    loss = jnp.linalg.norm(y-R(x), ord = 'fro')**2
    return loss

@jax.jit
def GD_step(x, y, R, alpha): 
    """
    Performs one step of gradient descent on the L2 loss.

    Parameters
    ----------
    x : jnp.ndarray
        Current estimate of the image.
    y : jnp.ndarray
        Measured sinogram (target).
    R : Callable
        Forward operator (e.g., Model_Beam_par).
    alpha : float
        Step size. 

    Returns
    -------
    loss_val : float
        Value of the loss function at current x.
    x_new : jnp.ndarray
        Updated estimated image after one gradient descent step.
    """
    loss_val, grads_val = jax.value_and_grad(loss_L2)(x, y, R)
    
    # GD update
    x_new = x - alpha * grads_val

    return loss_val, x_new


@jax.jit
def PGD_pos_supp_step(x, y, R, alpha): 
    """
    Performs one step of Proximal Gradient Descent (PGD) with
    support and positivity constraints.

    Parameters
    ----------
    x : jnp.ndarray
        Current estimate of the image.
    y : jnp.ndarray
        Measured sinogram (target).
    R : Callable
        Forward operator (e.g., Model_Beam_par).
    alpha : float
        Step size. 

    Returns
    -------
    loss_val : float
        Value of the loss function at current x.
    x_new : jnp.ndarray
        Updated estimated image after one PGD step.
    """
    
    loss_val, grads_val = jax.value_and_grad(loss_L2)(x, y, R)
    
    # GD update
    x_new = x - alpha * grads_val

    # support constraint
    x_new = x_new*disk_img
    
    # positivity constraint 
    x_new = x_new.clip(0, None)

    return loss_val, x_new



@jax.jit
def loss_L2_rL2(x, y, R, lam): 
    """
    Computes the L2 loss (LS) with L2 (Tikhonov) regularization.

    Parameters
    ----------
    x : jnp.ndarray
        Current estimate of the image.
    y : jnp.ndarray
        Measured sinogram (target).
    R : Callable
        Forward operator (e.g., Model_Beam_par).
    lam : float
        Regularization weight.

    Returns
    -------
    loss : float
        Tikhonov regularized L2 loss value.
    """
    loss = jnp.linalg.norm(y-R(x), ord = 'fro')**2 + lam*jnp.linalg.norm(x)**2
    return loss

@jax.jit
def PGD_pos_supp_rL2_step(x, y, R, alpha, lam): 
    """
    Performs one proximal gradient descent step with Tikhonov (L2) regularization, 
    positivity and support constraints.

    Parameters
    ----------
    x : jnp.ndarray
        Current estimate of the image.
    y : jnp.ndarray
        Measured sinogram (target).
    R : Callable
        Forward operator (e.g., Model_Beam_par).
    alpha : float
        Step size.
    lam : float
        Regularization weight.

    Returns
    -------
    loss_val : float
        Value of the loss function at current x.
    x_new : jnp.ndarray
        Updated estimated image after one PGD step.
    """
    
    loss_val, grads_val = jax.value_and_grad(loss_L2_rL2)(x, y, R, lam)
    
    # GD update
    x_new = x - alpha * grads_val

    # support constraint
    x_new = x_new*disk_img

    # positivity constraint 
    x_new = x_new.clip(0, None)
    
    return loss_val, x_new


@jax.jit
def TV(x): 
    """
    Compute the discrete Total Variation (TV) norm of a 2D image.

    The TV is approximated using forward finite differences along 
    both horizontal (x) and vertical (y) directions. The boundary 
    gradients are set to zero due to the forward difference scheme.

    Parameters
    ----------
    x : jnp.ndarray
        2D input image array.

    Returns
    -------
    float
        The total variation norm.
    """

    # use forward difference to compute grad_x and grad_y 
    grad_x = jnp.roll(x, -1, axis=1) - x
    grad_x = grad_x.at[:,-1].set(0) # (due to forward differences approx)

    grad_y = jnp.roll(x, -1, axis=0) - x
    grad_y = grad_y.at[-1].set(0) # (due to forward differences approx)

    # compute l1 norm of grad(.) 
    gx= jnp.sum(jnp.abs(grad_x)) + jnp.sum(jnp.abs(grad_y))
    return gx


@jax.jit
def loss_rTV(x, y, R, lam):
    """
    Compute the combined loss: L2 data fidelity term (LS) + weighted Total Variation regularization.

    Parameters
    ----------
    x : jnp.ndarray
        Current estimate of the image.
    y : jnp.ndarray
        Measured sinogram (target).
    R : callable
        Forward operator (e.g., Model_Beam_par).
    lam : float
        Regularization weight.

    Returns
    -------
    float
        Value of the regularized TV loss.
    """
    Fx = loss_L2(x, y, R) + lam* TV(x)
    return Fx


@jax.jit
def PGD_pos_supp_rTV_step(x, y, R, alpha, lam):  
    """
    Performs one proximal gradient descent step with TV regularization, 
    positivity and support constraints.

    Parameters
    ----------
    x : jnp.ndarray
        Current estimate of the image.
    y : jnp.ndarray
        Measured sinogram (target).
    R : Callable
        Forward operator (e.g., Model_Beam_par).
    alpha : float
        Step size.
    lam : float
        Regularization weight.

    Returns
    -------
    loss_val : float
        Value of the loss function at current x.
    x_new : jnp.ndarray
        Updated estimated image after one PGD step.
    """
    loss_val, grads_val = jax.value_and_grad(loss_rTV)(x, y, R, lam)
    
    # GD update
    x_new = x - alpha * grads_val

    # support constraint
    x_new = x_new*disk_img

    # positivity constraint 
    x_new = x_new.clip(0, None)

    return loss_val, x_new  


class Optimizer:
    """
    Class to perform iterative optimization for model based tomographic reconstruction.

    Attributes
    ----------
    angles : jnp.ndarray
        Array of projection angles in degrees.
    x : jnp.ndarray
        Current estimate of the image.
    y : jnp.ndarray
         Measured sinogram (target).
    R : Callable
        Forward operator (e.g., Model_Beam_par).
    solver : callable
        Optimization step function performing one update of x.
    alpha : float
        Step size.
    lam : float or None
        Regularization weight, if applicable.
    n_iter : int
        Number of iterations to run the optimization.

    Constructor parameters 
    ----------
    x_init : array-like
        Initial guess for the reconstructed image.
    y : array-like
        Measured sinogram (target).
    nbr_angles : int
        Number of projection angles between [0,180) degrees.
    solver : callable
        Optimization step function. Should have signature
        `solver(x, y, R, alpha)` or `solver(x, y, R, alpha, lam)`.
    alpha : float, optional
        Step size for the solver. Default is 1e-5.
    lam : float or None, optional
        Regularization weight passed to the solver if applicable. Default is None.
    n_iter : int, optional
        Number of iterations to run. Default is 360.

    Methods
    -------
    solve()
        Runs the iterative optimization for `n_iter` steps and returns the list of losses and final reconstruction.
    """
    angles: jnp.ndarray 
    x: jnp.ndarray
    y: jnp.ndarray
    R: Model_Beam_par
    solver: callable
    alpha: float
    lam: float 
    n_iter: int  


    def __init__(self, x_init, y, nbr_angles, solver,  alpha = 1e-5, lam = None, n_iter=360):
        # forward pass parameters
        angle_space = 180/nbr_angles
        self.angles = jnp.arange(0,180, angle_space)  
        self.R = Model_Beam_par(self.angles)

        # optimization parameters
        self.x = jnp.array(x_init)
        self.y = jnp.array(y) 
        self.solver = solver
        self.alpha = alpha
        self.lam = lam
        self.n_iter = n_iter

        # init solver
        if self.lam is None:
            _ = self.solver(self.x, self.y, self.R, self.alpha) 
        else:
            _ = self.solver(self.x, self.y, self.R, self.alpha, self.lam) 

    def solve(self):
        losses = []
        x = self.x
        for _ in tqdm(range(self.n_iter)):
            if self.lam is None:
                loss_val, x = self.solver(x, self.y, self.R, self.alpha) 
            else: 
                loss_val, x = self.solver(x, self.y, self.R, self.alpha, self.lam) 
            losses.append(loss_val)
        
        return losses, x
    

    
##################################################################################################################################################
# Metrics
def SNR(s, s_hat, normalize = True):
    """
    Compute the Signal-to-Noise Ratio (SNR) between a reference image and a reconstructed (or noisy) version.

    Parameters
    ----------
    s : ndarray
        Ground truth (reference) image.
    s_hat : ndarray
        Estimated or reconstructed image to compare with the reference.
    normalize : bool, default = True 
        input images are normalized to [0, 1] before computing the SNR

    Returns
    -------
    snr : float
        Signal-to-Noise Ratio in decibels (dB).
    """
    if normalize:
        s = normalize_img(s)
        s_hat = normalize_img((s_hat))
    snr = 10*np.log10(np.sum(s**2)/np.sum((s - s_hat)**2))
    return snr

def CNR(img, highlight = False):
    """
    Compute the Contrast-to-Noise Ratio (CNR) for predefined foreground/background regions in a normalized image.

    The CNR is computed for three pairs of regions of interest (foreground/background).
    Optionally, the selected areas can be highlighted in the output image.

    Parameters
    ----------
    img : array-like
        Input 2D image array.

    highlight : bool, optional (default is False)
        If True, return a copy of the image with selected regions highlighted (value set to 1).

    Returns
    -------
    list of float or tuple
        If `highlight` is False:
            List of CNR values for the three foreground/background region pairs.
        If `highlight` is True:
            Tuple `(img_norm, CNR_list)` where `img_norm` is the normalised image with selected regions set to 1,
            and `CNR_list` is the list of CNR values.
    
    Notes
    -----
    - The image is first normalized to [0, 1] using `normalize_img()`.
    - Foreground and background coordinates are hardcoded for each of the three cases.
    - The noise is estimated as the standard deviation in the background region.
    """
    CNR_list = []
    size = 10
    img_norm = np.array(normalize_img(img))
    for case in [0,1,2]:
        # select index area
        if case == 0:
            # foreground
            x1 = 95
            y1 = 190

            # background
            x2 =115
            y2 = 175

        if case == 1:
            # foreground
            x1 = 222
            y1 = 292

            # background
            x2 = 237
            y2 = 264

        if case == 2:
            # foreground
            x1 = 205
            y1 = 225

            # background
            x2 = 222
            y2 = 204

        
        # compute CNR
        area1 = img_norm[y1:y1+size, x1:x1+size]
        mean_area1 = area1.mean()
        
        area2 = img_norm[y2:y2+size, x2:x2+size]
        mean_area2 = area2.mean()
        
        std_noise = area2.std()
        
        CNR = np.abs(mean_area1 - mean_area2) / std_noise
        CNR_list.append(CNR)

        if highlight:
            # highlight areas 
            img_norm[y1:y1+size, x1:x1+size] = 1
            img_norm[y2:y2+size, x2:x2+size] = 1
        
    if highlight:
        return img_norm, CNR_list


    return CNR_list




##################################################################################################################################################
# Formatting results 
def normalize_img(img):
    """
    Normalize an image to the range [0, 1].

    Parameters
    ----------
    img : ndarray
        Input image to be normalized. Values below 0 are clipped.

    Returns
    -------
    img_normalized : ndarray
        Image with values scaled to the range [0, 1].
    """
    img_normalized = img.clip(0, None) 
    img_normalized = (img_normalized - img_normalized.min()) / jnp.ptp(img_normalized)
    return img_normalized 

def quant_res(target_img, rcst):
    """
    Utility function designed to simplify the notebook by printing quantitative 
    image metrics (CNR and SNR) between the target and reconstructed images
    """
    CNR_list = CNR(rcst) 
    SNR_val = SNR(target_img, rcst)
    print(f'CNR values: {CNR_list}, mean: {np.mean(CNR_list)}')
    print(f'SNR : {SNR_val} dB')
  
def summary_quant_res(target_img, rcst_FBP, rcst_SP, rcst_ML, noise_case):
    """
    Utility function created to simplify the notebook by grouping the display 
    of quantitative results (SNR and CNR) for different reconstruction methods.
    """
    print(f'{noise_case} noise level:')
    print('Only FBP')
    quant_res(target_img, rcst_FBP)
    print(20*'-')
    print('SP approach')
    quant_res(target_img, rcst_SP)
    print(20*'-')
    print('ML approach')
    quant_res(target_img, rcst_ML)
    if noise_case == 'Low':
        print(60*'-')



##################################################################################################################################################
# Dataset generation
def gen_img(shape, pattern, nbr_patterns ,seed, texture_scale, show = True): 
    """
    Generate 2D grayscale image with randomly oriented textured pattern.

    Parameters
    ----------
    shape: tuple[int, int]
        (height, width) Shape of the img in inches. 1 inches will be represented by 100 pixels
    pattern: str
        Choice of pattern in the img. Can be either 'ellipse', 'rectangle', 'triangle' or 'mix'
    nbr_patterns: int
        nbr of patterns to add inside the img
    seed: int
        seed for the reproducibility
    texture_scale: int or str
        texture_scale = 0 means no texture, texture_scale = 1 means fully textured. Also possible to have a mix of texture scale by setting texture_scale='mix'
    show: bool = True
        to display the resulting img

    Returns
    -------
    img: ndarray
        (height, width)
    """
    # init empty img
    fig, ax = plt.subplots(figsize = shape, dpi = 100)
    width, height = fig.get_size_inches()*fig.dpi
    img = np.zeros((int(width),int(height)))
    #add patterns
    mix_choice = None
    rng = np.random.default_rng(seed)  
    rng_texture = np.random.default_rng(seed + 10)  
    for i in range(nbr_patterns):
        # define pattern
        center_x = rng.uniform(0.1*height, height-0.1*height)
        center_y = rng.uniform(0.1*width, width- 0.1*width)
        angle = rng.uniform(-np.pi,np.pi)
        if pattern == 'mix':
            mix_choice =  rng.integers(1,4)
 
        if pattern == 'ellipse' or mix_choice == 1: 
            radius_x, radius_y = rng.uniform(5, 41, 2)
            rr, cc = ski.draw.ellipse(center_x, center_y, radius_x ,radius_y, shape= (int(width), int(height)), rotation= angle)

        if pattern == 'rectangle' or mix_choice == 2:
            rect_width = rng.uniform(10, 70)
            rect_height = rng.uniform(10, 50)
            dx = rect_width/2
            dy = rect_height/2
            corners = np.array([[-dx,-dy], [dx, -dy], [dx, dy], [-dx, dy]])

            # Rotate corners
            R = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle),  np.cos(angle)]])
            corners = corners@R + (center_x, center_y)

            rr, cc = ski.draw.polygon(corners[:, 0], corners[:, 1], shape= (int(width), int(height)))
        if pattern == 'triangle'  or mix_choice == 3:
            tri_width = rng.uniform(20, 70)
            tri_height = rng.uniform(20, 50)
            dx =  tri_width/2
            third_point_x = rng.uniform(-2*tri_height, 2*tri_height)
            corners = np.array([[-dx,0], [dx, 0], [third_point_x, tri_height]])

            
            # Rotate corners
            R = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle),  np.cos(angle)]])
            corners = corners@R + (center_x, center_y)

            rr, cc = ski.draw.polygon(corners[:, 0], corners[:, 1], shape= (int(width), int(height)))

        
        # Generate textured pattern
        if texture_scale == 'mix':
            scale = rng_texture.uniform(0,1)
        else:
            scale = texture_scale
        
        texture = 1 - rng_texture.uniform(0, 1, img.shape)*scale
        
        # Smoothen the img slightly to make cells look more natural
        sigma_0 = rng_texture.uniform(1,2)
        sigma_1 = rng_texture.uniform(1,2)
        texture = sc.ndimage.gaussian_filter1d(texture, sigma=sigma_0, axis = 0) 
        texture = sc.ndimage.gaussian_filter1d(texture, sigma=sigma_1, axis = 1) 

        # rotate smoothen texture to make img more natural
        angle = rng_texture.uniform(0, 180)
        texture = sc.ndimage.rotate(texture, angle, reshape=False)

        img[rr, cc] = texture[rr,cc]
            
    # circular clip img 
    disk_img = np.zeros(img.shape)
    disk_coord = ski.draw.disk((img.shape[0]//2,img.shape[1]//2), min(img.shape)/2*0.9)
    disk_img[disk_coord] = 1
    img = img*disk_img


    # Normalize
    img = (img - img.min()) / np.ptp(img)

    if show: 
        ax.imshow(img, cmap= 'gray')
        plt.show()
    return img 


def gen_dataset(shape, pattern, nbr_img, output_dir):
    """
    Generate Dataset of 2D grayscale image created from gen_img(...) function. Save resulting images in RGBA in .png format 

    Parameters
    ----------
    shape: tuple[int, int]
        (height, width) Shape of the img in inches. 1 inches will be represented by 100 pixels
    pattern: str
        Choice of pattern in the img. Can be either 'ellipse', 'rectangle', 'triangle', 'mix', or 'all'. pattern = 'all' will save images from all categories 
    nbr_img: int
        nbr of images to create for the chosen pattern. Note that if pattern = 'all', it will create nbr_img images per pattern. 
    output_dir: str
        directory where images will be saved
    Returns
    -------
    None. Images are saved in output_dir in RGBA in .png format
    """
    # create dir to save images
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = 1
    # uniform texture scale 
    list_texture_scale = np.linspace(0,1,nbr_img)
    for i in range(nbr_img):
        if pattern == 'all':
            pattern_list = ['rectangle', 'ellipse', 'triangle', 'mix']

        elif  pattern == 'triangle':
            pattern_list = ['triangle']

        elif  pattern == 'rectangle':
            pattern_list = ['rectangle']

        elif  pattern == 'ellipse':
            pattern_list = ['ellipse']
        
        elif  pattern == 'mix':
            pattern_list = ['mix']

        for pattern_choice in pattern_list:
            # create img
            print(f"Generating image {seed} over {len(pattern_list)*nbr_img} images", end='\r')
            img = gen_img(shape=shape, pattern=pattern_choice, nbr_patterns=100, seed=seed, texture_scale=list_texture_scale[i], show= False)
            
            # savec img
            img_name = pattern_choice + '_' + str(i) + '.png'
            plt.imsave(output_dir/img_name, img, cmap="gray")
            seed +=1