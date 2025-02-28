# imports 
import numpy as np

# gaussian kernel of shape (2k+1)*(2k+1) 
def get_2D_Gaussian_kernel(sigma_h):
    k = int(np.ceil(3*sigma_h))
    # 1D kernel
    kernel_1D = np.exp(-np.arange(-k, k+1)**2 / (2 * sigma_h**2))
    # normalisation
    kernel_1D /= kernel_1D.sum() 
    # 2D kernel
    kernel_2D = kernel_1D.reshape(-1, 1) @ kernel_1D.reshape(1, -1)
    return kernel_2D

# transform kernel (of shape (2k+1) x (2k+1)) to have same shape of the img to which it's applied with periodic boundary condition
def kernel_to_image(kernel, img):
    img_kernel = np.zeros(img.shape)
    k = kernel.shape[0] // 2
    img_kernel[:(2*k+1), :(2*k+1)] = kernel
    img_kernel = np.roll(img_kernel, (-k, -k), axis=(0, 1))
    return img_kernel
