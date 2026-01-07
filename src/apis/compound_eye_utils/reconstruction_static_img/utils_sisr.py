import torch
import torch.nn.functional as F
import numpy as np

# Helper functions adapted for PyTorch

def get_rho_sigma(sigma=2.55/255, iter_num=15, modelSigma1=49.0, modelSigma2=2.55, w=1.0):
    '''
    One can change the sigma to implicitly change the trade-off parameter
    between fidelity term and prior term
    '''
    modelSigmaS = np.logspace(np.log10(modelSigma1), np.log10(modelSigma2), iter_num).astype(np.float32)
    modelSigmaS_lin = np.linspace(modelSigma1, modelSigma2, iter_num).astype(np.float32)
    sigmas = (modelSigmaS*w+modelSigmaS_lin*(1-w))/255.
    rhos = list(map(lambda x: 0.23*(sigma**2)/(x**2), sigmas))
    return rhos, sigmas

def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    This function computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.

    Args:
        psf (torch.Tensor): The point-spread function (kernel). 
                            Expected shape: (1, 1, h, w)
        shape (tuple): The target shape [H, W] for the OTF.

    Returns:
        torch.Tensor: The optical transfer function (OTF).
    '''
    # Create a tensor of the target shape, initially all zeros.
    otf = torch.zeros(psf.shape[0], psf.shape[1], shape[0], shape[1], dtype=psf.dtype, device=psf.device)
    
    # Copy the PSF into the top-left corner of the larger tensor.
    otf[..., :psf.shape[2], :psf.shape[3]].copy_(psf)
    
    # Roll the PSF to the center to account for the shift before FFT.
    # This is crucial for correct frequency domain representation.
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis + 2)
        
    # Compute the 2D FFT of the shifted PSF.
    # The result is a complex tensor.
    otf = torch.fft.fft2(otf, dim=(-2, -1))
    
    return otf


def upsample(x, sf=3):
    '''
    S-fold upsampler using nearest-neighbor interpolation followed by zero-filling.
    This is the transpose of the downsampling operation.

    Args:
        x (torch.Tensor): The input low-resolution tensor image. 
                          Shape: (1, 1, h, w)
        sf (int): The scaling factor.

    Returns:
        torch.Tensor: The upsampled tensor. Shape: (1, 1, H, W)
    '''
    # Use interpolate for a clean way to upsample and then create the sparse matrix
    # by multiplying with a mask. This is equivalent to filling new entries with zeros.
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*sf, x.shape[3]*sf), dtype=x.dtype, device=x.device)
    z[..., st::sf, st::sf].copy_(x)
    return z


# Main Pre-computation Function

def pre_compute_all_frames(lr_imgs, kernels, sf):
    """
    Performs all pre-computations needed for the MF-PnP algorithm.

    Args:
        lr_imgs (list[torch.Tensor]): List of 10 low-resolution 2D grayscale images.
        kernels (list[torch.Tensor]): List of 10 2D filtering kernels.
        sf (int): The integer scaling factor for super-resolution.

    Returns:
        tuple: A tuple containing:
            - z0 (torch.Tensor): The initial estimate for the high-resolution image.
            - FBs (list[torch.Tensor]): List of Fourier transforms of the kernels (F(k)).
            - FBCs (list[torch.Tensor]): List of complex conjugates of FBs.
            - FBFys (list[torch.Tensor]): List of FBC * F(D^T*y).
    """
    print("Starting pre-computation...")

    # 1. Initialize z0: The first estimate of the HR image
    # We use the first LR image and upscale it using bicubic interpolation.
    # This provides a reasonable starting point for the iterative algorithm.
    # Ensure the input tensor is 4D for F.interpolate
    ref_lr_img_4d = lr_imgs[0].unsqueeze(0).unsqueeze(0) 
    hr_h = ref_lr_img_4d.shape[2] * sf
    hr_w = ref_lr_img_4d.shape[3] * sf
    
    z0 = F.interpolate(ref_lr_img_4d, size=(hr_h, hr_w), mode='bicubic', align_corners=False)
    z0 = z0.squeeze(0).squeeze(0) # Back to 2D
    print(f"Initialized z0 with shape: {z0.shape}")

    # Lists to store the pre-computed values for each of the 10 frames
    FBs = []
    FBCs = []
    FBFys = []

    # 2. Loop through each frame to perform pre-calculation
    for i in range(len(lr_imgs)):
        lr_img = lr_imgs[i].unsqueeze(0).unsqueeze(0) # Add batch and channel dims
        kernel = kernels[i].unsqueeze(0).unsqueeze(0) # Add batch and channel dims

        # FB: The Optical Transfer Function (OTF) of the kernel, F(k)
        FB = p2o(kernel, (hr_h, hr_w))
        FBs.append(FB)

        # FBC: The complex conjugate of FB, F(k)_bar
        FBC = torch.conj(FB)
        FBCs.append(FBC)

        # FBFy: FBC * F(D^T*y)
        # First, perform the transpose-downsample (upsample with zero-filling)
        STy = upsample(lr_img, sf=sf)
        # Then, compute its FFT
        F_STy = torch.fft.fft2(STy, dim=(-2, -1))
        # Finally, multiply with the conjugate kernel
        FBFy = FBC * F_STy
        FBFys.append(FBFy)
        
        print(f"Processed frame {i+1}/{len(lr_imgs)}")

    print("Pre-computation finished.")
    return z0, FBs, FBCs, FBFys

def pre_calculate_old(x, k, sf):
    '''
    Args:
        x: NxCxHxW, LR input
        k: NxCxhxw, 2d kernel
        sf: integer, scaling factor

    Returns:
        FB, FBC, F2B, FBFy
        will be reused during iterations
    '''
    w, h = x.shape[-2:]
    FB = p2o(k, (w*sf, h*sf))
    FBC = torch.conj(FB)
    F2B = torch.pow(torch.abs(FB), 2)
    STy = upsample(x, sf=sf)
    FBFy = FBC*torch.fft.fftn(STy, dim=(-2, -1))
    return FB, FBC, F2B, FBFy

# -----------------------------------------------------------------------------
# Functions for the Iterative Data Reconstruction Step
# -----------------------------------------------------------------------------

def downsample_fourier(x, sf=3):
    """
    Performs the Fourier domain downsampling operation (S).
    This is equivalent to averaging distinct sfxsf blocks in the spatial domain.
    We implement it efficiently using average pooling on the complex tensor.
    
    Args:
        x (torch.Tensor): Input complex tensor in the Fourier domain.
        sf (int): The scaling factor.

    Returns:
        torch.Tensor: The downsampled complex tensor.
    """
    # Average pooling works on real tensors, so we handle real and imaginary parts separately
    return torch.complex(
        F.avg_pool2d(x.real, kernel_size=sf, stride=sf, padding=0),
        F.avg_pool2d(x.imag, kernel_size=sf, stride=sf, padding=0)
    )

def upsample_fourier(x, sf=3):
    """
    Performs the Fourier domain upsampling operation (S^T).
    This is the transpose of the downsampling operation, implemented efficiently
    using nearest-neighbor interpolation.

    Args:
        x (torch.Tensor): Input complex tensor in the downsampled Fourier domain.
        sf (int): The scaling factor.
        
    Returns:
        torch.Tensor: The upsampled complex tensor.
    """
    return torch.complex(
        F.interpolate(x.real, scale_factor=sf, mode='nearest'),
        F.interpolate(x.imag, scale_factor=sf, mode='nearest')
    )

def splits(a, sf):
    '''split a into sfxsf distinct blocks
    Args:
        a: NxCxWxH
        sf: split factor
    Returns:
        b: NxCx(W/sf)x(H/sf)x(sf^2)
    '''
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=4)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=4)
    return b



def multi_frame_data_solution(z_k, FBs, FBCs, FBFys, mu, sf):
    """
    Solves the multi-frame data subproblem for a single iteration.
    This is the core of the reconstruction step, generalizing the single-frame
    Wiener deconvolution to handle a burst of images with different kernels.

    Args:
        z_k (torch.Tensor): The output from the denoiser in the previous step (2D or 4D).
        FBs (list[torch.Tensor]): Pre-computed list of Fourier kernels F(K_i).
        FBCs (list[torch.Tensor]): Pre-computed list of conjugate Fourier kernels.
        FBFys (list[torch.Tensor]): Pre-computed list of FBC_i * F(D^T*y_i).
        mu (float): The penalty parameter for the current iteration.
        sf (int): The scaling factor.

    Returns:
        torch.Tensor: The updated high-resolution image estimate, x_{k+1}.
    """
    # Ensure z_k is in the 4D format (B, C, H, W) for FFT
    if z_k.dim() == 2:
        z_k_4d = z_k.unsqueeze(0).unsqueeze(0)
    else:
        z_k_4d = z_k

    # Pre-stage:
    kernel_num = len(FBs)

    total_FX = torch.zeros_like(FBs[0])

    for i in range(kernel_num):
        # 1. Calculate the sum of |F(K_i)|^2 for one frame
        F2B = torch.pow(torch.abs(FBs[i]), 2)

        # 2. Calculate the main numerator term, FR = sum(FBFy_i) + mu*F(z_k)
        FBFy = FBFys[i]
        F_z_k = torch.fft.fft2(z_k_4d, dim=(-2, -1))
        FR = FBFy + mu * F_z_k

        # 3. Calculate the denominator of the correction term
        # This involves the Fourier downsampling operation S
        # den = downsample_fourier(sum_F2B, sf) + mu
        F2B_real = torch.pow(torch.abs(FBs[i]), 2)
        # den = F.avg_pool2d(sum_F2B_real, kernel_size=sf, stride=sf, padding=0) + mu
        
        den = torch.mean(splits(F2B_real, sf), dim=-1, keepdim=False) + mu

        # 4. Calculate the total correction term by summing contributions from each frame

        # Numerator for this frame's correction term
        # num_i = downsample_fourier(FBs[i] * FR, sf)
        num_i = torch.mean(splits(FBs[i] * FR, sf), dim=-1, keepdim=False)
        
        # The fraction inside the correction term
        term_i = num_i / den
        
        # Apply Fourier upsampling (S^T) and multiply by the conjugate kernel
        # correction_i = FBCs[i] * upsample_fourier(term_i, sf)
        correction_i = FBCs[i] * term_i.repeat(1, 1, sf, sf)

        # 5. Calculate the final solution for F(x_{k+1}) in the Fourier domain
        FX = (FR - correction_i) / mu

        total_FX += FX

    total_FX /= kernel_num
    
    # 6. Inverse FFT to get the spatial domain image and return the real part
    x_k_plus_1 = torch.fft.ifft2(total_FX, dim=(-2, -1)).real

    # Return in the same dimension as the input z_k
    return x_k_plus_1 if z_k.dim() == 4 else x_k_plus_1.squeeze(0).squeeze(0)


def data_reconstruction_otf_api(x, args_dict):
    """
    API wrapper for the multi-frame data solution step.

    Args:
        x (torch.Tensor): The temporal reconstructed high-resolution image from the
                          previous (denoiser) step.
        args_dict (dict): A dictionary containing all pre-computed tensors and parameters.
                          Expected keys: 'FBs', 'FBCs', 'FBFys', 'mu', 'sf'.

    Returns:
        torch.Tensor: Reconstructed high-resolution image for the current iteration.
    """
    # Unpack pre-computed variables and parameters from the dictionary
    FBs = args_dict["FBs"]
    FBCs = args_dict["FBCs"]
    FBFys = args_dict["FBFys"]
    mu = args_dict["mu"]
    sf = args_dict["sf"]

    # Call the core solver function
    reconstructed_x = multi_frame_data_solution(x, FBs, FBCs, FBFys, mu, sf)

    return reconstructed_x

# -----------------------------------------------------------------------------
# Functions for reconstructing different images
# -----------------------------------------------------------------------------

def pre_compute_all_frames_multi_frame(lr_imgs, kernels, sf):
    """
    Performs all pre-computations needed for the MF-PnP algorithm.

    Args:
        lr_imgs (list[torch.Tensor]): List of 10 low-resolution 2D grayscale images.
        kernels (list[torch.Tensor]): List of 10 2D filtering kernels.
        sf (int): The integer scaling factor for super-resolution.

    Returns:
        tuple: A tuple containing:
            - z0 ([torch.Tensor]): The initial estimate for the high-resolution image.
            - FBs (list[torch.Tensor]): List of Fourier transforms of the kernels (F(k)).
            - FBCs (list[torch.Tensor]): List of complex conjugates of FBs.
            - FBFys (list[torch.Tensor]): List of FBC * F(D^T*y).
    """
    print("Starting pre-computation...")

    # 1. Initialize z0: The first estimate of the HR image
    # We use the first LR image and upscale it using bicubic interpolation.
    # This provides a reasonable starting point for the iterative algorithm.
    # Ensure the input tensor is 4D for F.interpolate
    ref_lr_img_4d = lr_imgs[0].unsqueeze(0).unsqueeze(0) 
    hr_h = ref_lr_img_4d.shape[2] * sf
    hr_w = ref_lr_img_4d.shape[3] * sf
    
    z0_list = []
    for i in range(len(lr_imgs)):
        z0 = F.interpolate(lr_imgs[i].unsqueeze(0).unsqueeze(0) , size=(hr_h, hr_w), mode='bicubic', align_corners=False)
        z0 = z0.squeeze(0).squeeze(0) # Back to 2D
        z0_list.append(z0)

    # Lists to store the pre-computed values for each of the 10 frames
    FBs = []
    FBCs = []
    FBFys = []

    # 2. Loop through each frame to perform pre-calculation
    for i in range(len(lr_imgs)):
        lr_img = lr_imgs[i].unsqueeze(0).unsqueeze(0) # Add batch and channel dims
        kernel = kernels[i].unsqueeze(0).unsqueeze(0) # Add batch and channel dims

        # FB: The Optical Transfer Function (OTF) of the kernel, F(k)
        FB = p2o(kernel, (hr_h, hr_w))
        FBs.append(FB)

        # FBC: The complex conjugate of FB, F(k)_bar
        FBC = torch.conj(FB)
        FBCs.append(FBC)

        # FBFy: FBC * F(D^T*y)
        # First, perform the transpose-downsample (upsample with zero-filling)
        STy = upsample(lr_img, sf=sf)
        # Then, compute its FFT
        F_STy = torch.fft.fft2(STy, dim=(-2, -1))
        # Finally, multiply with the conjugate kernel
        FBFy = FBC * F_STy
        FBFys.append(FBFy)
        
        # print(f"Processed frame {i+1}/{len(lr_imgs)}")

    # print("Pre-computation finished.")
    return z0_list, FBs, FBCs, FBFys


def multi_frame_data_solution_multi_frame(z_k_list, FBs, FBCs, FBFys, mu, sf):
    """
    Solves the multi-frame data subproblem for a single iteration.
    This is the core of the reconstruction step, generalizing the single-frame
    Wiener deconvolution to handle a burst of images with different kernels.

    Args:
        z_k_list ([torch.Tensor]): List of the output from the denoiser in the previous step (2D or 4D).
        FBs (list[torch.Tensor]): Pre-computed list of Fourier kernels F(K_i).
        FBCs (list[torch.Tensor]): Pre-computed list of conjugate Fourier kernels.
        FBFys (list[torch.Tensor]): Pre-computed list of FBC_i * F(D^T*y_i).
        mu (float): The penalty parameter for the current iteration.
        sf (int): The scaling factor.

    Returns:
        torch.Tensor: The updated high-resolution image estimate, x_{k+1}.
    """
    # Ensure z_k is in the 4D format (B, C, H, W) for FFT

    # Pre-stage:
    kernel_num = len(FBs)

    res_xs = []

    for i in range(kernel_num):
        # Ensure z_k is in the 4D format (B, C, H, W) for FFT
        z_k = z_k_list[i]

        if z_k.dim() == 2:
            z_k_4d = z_k.unsqueeze(0).unsqueeze(0)
        else:
            z_k_4d = z_k

        # 1. Calculate the sum of |F(K_i)|^2 for one frame
        F2B = torch.pow(torch.abs(FBs[i]), 2)

        # 2. Calculate the main numerator term, FR = sum(FBFy_i) + mu*F(z_k)
        FBFy = FBFys[i]
        F_z_k = torch.fft.fft2(z_k_4d, dim=(-2, -1))
        FR = FBFy + mu * F_z_k

        # 3. Calculate the denominator of the correction term
        # This involves the Fourier downsampling operation S
        # den = downsample_fourier(sum_F2B, sf) + mu
        F2B_real = torch.pow(torch.abs(FBs[i]), 2)
        # den = F.avg_pool2d(sum_F2B_real, kernel_size=sf, stride=sf, padding=0) + mu
        
        den = torch.mean(splits(F2B_real, sf), dim=-1, keepdim=False) + mu

        # 4. Calculate the total correction term by summing contributions from each frame

        # Numerator for this frame's correction term
        # num_i = downsample_fourier(FBs[i] * FR, sf)
        num_i = torch.mean(splits(FBs[i] * FR, sf), dim=-1, keepdim=False)
        
        # The fraction inside the correction term
        term_i = num_i / den
        
        # Apply Fourier upsampling (S^T) and multiply by the conjugate kernel
        # correction_i = FBCs[i] * upsample_fourier(term_i, sf)
        correction_i = FBCs[i] * term_i.repeat(1, 1, sf, sf)

        # 5. Calculate the final solution for F(x_{k+1}) in the Fourier domain
        FX = (FR - correction_i) / mu

        x_k_plus_1 = torch.fft.ifft2(FX, dim=(-2, -1)).real

        res_xs.append(x_k_plus_1 if z_k.dim() == 4 else x_k_plus_1.squeeze(0).squeeze(0))

    # Return in the same dimension as the input z_k
    return res_xs


def data_reconstruction_otf_api_multi_frame(x, args_dict):
    """
    API wrapper for the multi-frame data solution step.

    Args:
        x ([torch.Tensor]): List of the temporal reconstructed high-resolution images from the
                          previous (denoiser) step.
        args_dict (dict): A dictionary containing all pre-computed tensors and parameters.
                          Expected keys: 'FBs', 'FBCs', 'FBFys', 'mu', 'sf'.

    Returns:
        [torch.Tensor]: list of reconstructed high-resolution image for the current iteration.
    """
    # Unpack pre-computed variables and parameters from the dictionary
    FBs = args_dict["FBs"]
    FBCs = args_dict["FBCs"]
    FBFys = args_dict["FBFys"]
    mu = args_dict["mu"]
    sf = args_dict["sf"]

    # Call the core solver function
    reconstructed_xs = multi_frame_data_solution_multi_frame(x, FBs, FBCs, FBFys, mu, sf)

    return reconstructed_xs

# -----------------------------------------------------------------------------
# Functions for the Denoising Step (Prior)
# -----------------------------------------------------------------------------

def test_split_fn(model, L, refield=32, min_size=256, sf=1, modulo=1):
    '''
    Denoise a large image by splitting it into overlapping patches.
    This is a recursive function.

    Args:
        model: The denoiser model (e.g., DRUNet).
        L: Input Low-quality image, with noise map concatenated as a channel.
        refield: Effective receptive field of the network.
        min_size: Minimum size of a patch to be processed directly.
        sf: Scale factor (1 for denoising).
        modulo: Padding modulo.
    Returns:
        torch.Tensor: The denoised image.
    '''
    h, w = L.size()[-2:]
    if h * w <= min_size**2:
        # If the image is small enough, process it directly
        # Pad to be divisible by modulo
        L = F.pad(L, (0, int(np.ceil(w/modulo)*modulo-w), 0, int(np.ceil(h/modulo)*modulo-h)), 'replicate')
        E = model(L)
        # Crop back to original size
        E = E[..., :h*sf, :w*sf]
    else:
        # If the image is large, split into 4 overlapping quadrants
        top = slice(0, (h//2//refield+1)*refield)
        bottom = slice(h - (h//2//refield+1)*refield, h)
        left = slice(0, (w//2//refield+1)*refield)
        right = slice(w - (w//2//refield+1)*refield, w)
        Ls = [L[..., top, left], L[..., top, right], L[..., bottom, left], L[..., bottom, right]]

        # Recursively process each quadrant
        Es = [test_split_fn(model, Ls[i], refield=refield, min_size=min_size, sf=sf, modulo=modulo) for i in range(4)]

        # Stitch the results back together
        b, c = Es[0].size()[:2]
        E = torch.zeros(b, c, sf * h, sf * w).type_as(L)

        E[..., :h//2*sf, :w//2*sf] = Es[0][..., :h//2*sf, :w//2*sf]
        E[..., :h//2*sf, w//2*sf:w*sf] = Es[1][..., :h//2*sf, (-w + w//2)*sf:]
        E[..., h//2*sf:h*sf, :w//2*sf] = Es[2][..., (-h + h//2)*sf:, :w//2*sf]
        E[..., h//2*sf:h*sf, w//2*sf:w*sf] = Es[3][..., (-h + h//2)*sf:, (-w + w//2)*sf:]
    return E

def denoise_step_api(x, model, sigma, refield=64, min_size=256, modulo=16):
    """
    API wrapper for the denoising step.

    Args:
        x (torch.Tensor): The noisy high-resolution image from the data step.
        model: The pre-trained denoiser network (e.g., DRUNet).
        sigma (float): The effective noise level for the current iteration.
    Returns:
        torch.Tensor: The denoised high-resolution image.
    """
    # Ensure input is 4D
    x_4d = x.unsqueeze(0).unsqueeze(0) if x.dim() == 2 else x
    
    # Create the noise map and concatenate it with the image
    # The denoiser expects a 2-channel input: [image, noise_map]
    noise_map = torch.full((x_4d.shape[0], 1, x_4d.shape[2], x_4d.shape[3]), sigma, device=x.device, dtype=x.dtype)
    x_with_noise_map = torch.cat((x_4d, noise_map), dim=1)
    
    # Denoise the image using the splitting strategy
    denoised_x = test_split_fn(model, x_with_noise_map, refield=refield, min_size=min_size, sf=1, modulo=modulo)
    
    # Return in the same dimension as the input x
    return denoised_x if x.dim() == 4 else denoised_x.squeeze(0).squeeze(0)


if __name__ == '__main__':
    # --- Create Dummy Data for Demonstration ---
    # This simulates the input variables you would have.
    
    SCALE_FACTOR = 4
    NUM_FRAMES = 10
    LR_IMG_SIZE = 64
    KERNEL_SIZE = 13

    # Use CUDA if available for faster FFT computations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. lr_imgs: List of 10 low-resolution images
    # Simulating random noise images for this example
    dummy_lr_imgs = [torch.rand(LR_IMG_SIZE, LR_IMG_SIZE, device=device) for _ in range(NUM_FRAMES)]

    # 2. kernels: List of 10 filtering kernels
    # Simulating random kernels
    dummy_kernels = [torch.rand(KERNEL_SIZE, KERNEL_SIZE, device=device) for _ in range(NUM_FRAMES)]
    # Normalize kernels so they sum to 1, which is typical for blur kernels
    dummy_kernels = [k / torch.sum(k) for k in dummy_kernels]

    # --- Run the Pre-computation ---
    z0, FBs, FBCs, FBFys = pre_compute_all_frames(dummy_lr_imgs, dummy_kernels, sf=SCALE_FACTOR)

    # --- Verify the Outputs ---
    print("\n--- Verification of Output Shapes ---")
    print(f"Shape of initial HR estimate z0: {z0.shape}")
    
    # The HR shape should be LR_IMG_SIZE * SCALE_FACTOR
    expected_hr_shape = (LR_IMG_SIZE * SCALE_FACTOR, LR_IMG_SIZE * SCALE_FACTOR)
    print(f"Expected HR shape: {expected_hr_shape}")
    assert z0.shape == expected_hr_shape
    
    print(f"\nNumber of FB tensors: {len(FBs)}")
    print(f"Shape of a single FB tensor: {FBs[0].shape}")
    # The shape should be (1, 1, HR_H, HR_W)
    expected_f_shape = (1, 1, expected_hr_shape[0], expected_hr_shape[1])
    assert FBs[0].shape == expected_f_shape
    
    print(f"\nNumber of FBC tensors: {len(FBCs)}")
    print(f"Shape of a single FBC tensor: {FBCs[0].shape}")
    assert FBCs[0].shape == expected_f_shape

    print(f"\nNumber of FBFy tensors: {len(FBFys)}")
    print(f"Shape of a single FBFy tensor: {FBFys[0].shape}")
    assert FBFys[0].shape == expected_f_shape
    
    print("\nAll output shapes are correct.")

