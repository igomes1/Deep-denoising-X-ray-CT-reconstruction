# imports 
import jax 
import numpy as np
import jax.numpy as jnp
import scipy as sc
import matplotlib.pyplot as plt
import skimage as ski
from pathlib import Path



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

@jax.jit
def rotate_image(image, angle):# angle in degree
    angle_rad = -angle * jnp.pi / 180  # Conversion en radians
    
    # Coordonnées des pixels
    H, W = image.shape
    x, y = jnp.meshgrid(jnp.arange(W), jnp.arange(H))
    x = x - W // 2
    y = y - H // 2
    
    # Matrice de rotation
    cos_theta = jnp.cos(angle_rad)
    sin_theta = jnp.sin(angle_rad)
    x_rot = cos_theta * x - sin_theta * y + W // 2
    y_rot = sin_theta * x + cos_theta * y + H // 2
    
    # Interpolation bilinéaire
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


####### should maybe normalize it before to return sinogram !!!
@jax.jit
def radon_transform(img, angles): # angle in degree

    dim_x, dim_y = img.shape
    diag = int(np.ceil(np.sqrt(dim_x**2 + dim_y**2)))  

    # pad img with zero due to rotation (max dimension that is projected = diag)
    padded_image = jnp.zeros((diag, diag))   
    
    # Center the img inside padded_img
    offset_x = (diag - dim_x) // 2
    offset_y = (diag - dim_y) // 2
    # padded_image[offset_x:offset_x + dim_x, offset_y:offset_y + dim_y] = img
    padded_image = padded_image.at[offset_x:offset_x + dim_x, offset_y:offset_y + dim_y].set(img)

   
    # projection
    sinogram = jnp.zeros((diag, len(angles)))  
    for i, angle in enumerate(angles):
        # rotated = jnp.array(sc.ndimage.rotate(padded_image, -angle, reshape=False))
        # rotated = dm_pix.rotate(jnp.expand_dims(padded_image, axis=-1), -angle*np.pi/180, order = 1, mode='constant')[:,:,0]
        rotated = rotate_image(padded_image, angle)
        
        sinogram = sinogram.at[:,i].set(rotated.sum(axis=1))

    # # normalize
    # sinogram = sinogram.clip(0, None) 
    # sinogram = (sinogram - sinogram.min()) / jnp.ptp(sinogram)

    return sinogram

# Define functions for reconstruction 

# Filtered back projection
def ramp_fourier_filter(size, filter_name =None): # took from sckit-image radon transform module
    n = np.concatenate(
        (
            np.arange(1, size / 2 + 1, 2, dtype=int),
            np.arange(size / 2 - 1, 0, -2, dtype=int),
        )
    )
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2

    fourier_filter = 2 * np.real(sc.fft.fft(f))  # ramp filter

    # if we want a modification of the filter --->from skimage too
    if filter_name == "cosine":
        freq = np.linspace(0, np.pi, size, endpoint=False)
        cosine_filter = sc.fft.fftshift(np.sin(freq)) 
        fourier_filter *= cosine_filter

    return fourier_filter[:, np.newaxis]
    

def nimp():
    print(3)

# Filter
def fbp_par(sinogramm, angles, output_size, subsampling = False, sub_sampled_angles = None, filter_name = None): # angles in degree
    
    # normelize sinogramm
    # sinogramm = (sinogramm - sinogramm.min()) / np.ptp(sinogramm) 
    
    sino_shape = sinogramm.shape[0]

    #resize sinogramm next power of two for fourrier analyses
    projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * sino_shape))))
    pad_width = ((0, projection_size_padded - sino_shape), (0, 0))
    sinogramm_padded = np.pad(sinogramm, pad_width, mode='constant', constant_values=0)


    # apply Ramp filter 
    fourier_filter = ramp_fourier_filter(projection_size_padded, filter_name)
    projection = sc.fft.fft(sinogramm_padded, axis=0) * fourier_filter
    sinogramm_filtered = np.real(sc.fft.ifft(projection, axis=0)[:sino_shape, :])
    # sinogramm_filtered = sinogramm


    # back projection 
    recons_img = np.zeros((output_size, output_size))
    center = output_size//2
    xpr, ypr = np.mgrid[:output_size, :output_size] - center
    x = np.arange(sino_shape) - sino_shape // 2 # t
    for col, angle in zip(sinogramm_filtered.T, angles):
        if not subsampling:
            t = xpr * np.cos(angle*np.pi/180) + ypr * np.sin(angle*np.pi/180) # t here is a matrix where the value of t[x,y] = t(x,y)

            # use linear interpolation
            # print(np.interp(t, xp=x, fp=col, left=0, right=0).shape)
            recons_img += np.interp(t, xp=x, fp=col, left=0, right=0)
        else:
            if angle in sub_sampled_angles:
                t = xpr * np.cos(angle*np.pi/180) + ypr * np.sin(angle*np.pi/180) # t here is a matrix where the value of t[x,y] = t(x,y)

                # use linear interpolation
                # print(np.interp(t, xp=x, fp=col, left=0, right=0).shape)
                recons_img += np.interp(t, xp=x, fp=col, left=0, right=0)

    return recons_img*np.pi/(2*len(angles))
    # return recons_img

def SNR(s, s_hat):
    snr = 10*np.log10(np.sum(s**2)/np.sum((s - s_hat)**2))
    return snr

def normalize_img(img):
    img = img.clip(0, None) 
    img = (img - img.min()) / jnp.ptp(img)
    return img 


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
