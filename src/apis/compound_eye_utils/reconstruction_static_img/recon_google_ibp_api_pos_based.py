import torch
import torch.nn.functional as F
import numpy as np
from scipy.special import erf
import cv2
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import convolve
from scipy.interpolate import griddata
from scipy.signal import correlate2d
import matplotlib.pyplot as plt

from .utils_kernel import *
from .mf_pnp_reconstruction import mf_pnp_reconstruction
from .ibp.google_ibp import iterative_back_projection

def google_ibp_api(lr_imgs_torch, kernels_torch, sf=4, K_max=24, mode="with_pnp", initial_hr_estimation: torch.Tensor=None):
    
    if mode == "with_pnp" or mode == "only_pnp":
        estimated_hr_torch = mf_pnp_reconstruction(lr_imgs_torch, kernels_torch, sf, K_max=K_max, initial_hr_estimation=initial_hr_estimation)
    else:
        ref_lr_img = lr_imgs_torch[-1].numpy()
        new_size = (int(ref_lr_img.shape[0] * sf), int(ref_lr_img.shape[1] * sf))
        bicubic_hr = cv2.resize(ref_lr_img, new_size, interpolation=cv2.INTER_CUBIC)
        estimated_hr_torch = torch.from_numpy(bicubic_hr)

    if mode == "only_pnp":
        reconstructed_hr_torch = estimated_hr_torch
    else:
        reconstructed_hr_torch = iterative_back_projection(lr_imgs_torch, kernels_torch, estimated_hr_torch.to("cuda"), 64, sf)
    
    return reconstructed_hr_torch


def pure_ibp_api(estimated_hr_torch, lr_imgs_torch, kernels_torch, sf=4, iter_times=64):
    reconstructed_hr_torch = iterative_back_projection(lr_imgs_torch, kernels_torch, estimated_hr_torch.float().to("cuda"), iter_times, sf)
    return reconstructed_hr_torch


def place_pixel_based_on_pos_rcd(hr_template, sampled_img, pos_arr):
    """
    placing the pixel values in the sampled_img to hr_template based on pos_arr
    (Faster version using NumPy advanced indexing)

    :param hr_template: (H, W) high resolution array template
    :param sampled_img: (omm_num_sqrt, omm_num_sqrt)
    :param pos_arr:  ( omm_num_sqrt ** 2, 2) x, y position for sample_img pixels on the high resolution image
    :return:
    """
    # Ensure all inputs are NumPy arrays for indexing
    hr_template = np.asarray(hr_template)
    sampled_img = np.asarray(sampled_img)
    pos_arr = np.asarray(pos_arr)

    # 1. Prepare the indices for the hr_template
    # pos_arr contains the (x, y) coordinates for the hr_template.
    # The x coordinates (rows) are in the first column (index 0).
    # The y coordinates (columns) are in the second column (index 1).
    # It's crucial to ensure these are integer types for indexing.

    col_indices = pos_arr[:, 0].astype(int)
    row_indices = pos_arr[:, 1].astype(int)

    # 2. Prepare the pixel values to place
    # The pixels in sampled_img are mapped sequentially based on pos_arr order.
    # The flattened sampled_img contains the values in the order needed.
    # Note: sampled_img is (N, N), and sampled_img.flatten() is (N*N).

    pixel_values = sampled_img.flatten()

    # 3. Use advanced indexing (vectorized operation)
    # This assigns all pixel_values to the specified (row, col) positions in hr_template simultaneously.
    hr_template[row_indices, col_indices] = pixel_values

    return hr_template


def interpolate_reconstruct_hr_image(sampled_imgs, pos_rcd, src_img_size, up_scale=4, method="nearest"):
    """
    Robust reconstruction using Direct Interpolation.

    Args:
        sampled_imgs: (3, H, W) array of pixel values.
        pos_rcd: (3, H, W, 2) array of coordinates (y, x) corresponding to src_img_size.
        src_img_size: The dimension of the coordinate system used in pos_rcd.
        up_scale: The target scaling factor relative to the low-res input.

    Returns:
        torch.Tensor: The reconstructed HR image.
    """
    print(f" [Recon] Running Direct Interpolation (Scale: {up_scale}x)")

    # 1. Flatten the Data
    # We treat all pixels from all 3 images as a single "cloud" of points.
    # values: (N,)
    values = sampled_imgs.reshape(-1)

    # points: (N, 2) -> These are (y, x) coordinates in the 'src_img_size' domain
    points = pos_rcd.reshape(-1, 2)

    # 2. Calculate the Scaling Factor (No Rounding Errors)
    # We want the output to be (lr_height * up_scale).
    lr_height, lr_width = sampled_imgs[0].shape
    target_h = int(lr_height * up_scale)
    target_w = int(lr_width * up_scale)

    # Calculate how much we need to scale the coordinates to fit the new target box
    # If pos_rcd is normalized to src_img_size, we scale it to target_h/w.
    scale_y = target_h / float(src_img_size)
    scale_x = target_w / float(src_img_size)

    # Apply scaling directly to the float coordinates
    # This preserves sub-pixel precision (e.g. 50.5 -> 202.0)
    scaled_points = np.copy(points)
    scaled_points[:, 0] *= scale_y
    scaled_points[:, 1] *= scale_x

    # 3. Define the Target Grid
    # We create a grid for exactly the pixels we want to output.
    # No "cropping" needed - we just don't ask for pixels outside this box.
    grid_y, grid_x = np.mgrid[0:target_h, 0:target_w]

    # 4. Interpolate
    # Since your data is concentrated in the "central 0.9", the edges of this
    # grid will automatically be filled by the nearest valid pixel (extrapolation)
    # or left as valid data if the FOV matches.

    # 'nearest' is fast and ensures no NaNs.
    # 'cubic' is smoother but requires handling NaNs at the edges (see below).
    reconstructed_image = griddata(
        scaled_points,
        values,
        (grid_y, grid_x),
        method=method
    )

    # (Optional) If using 'cubic', handle the empty edges:
    if np.any(np.isnan(reconstructed_image)):
        mask = np.isnan(reconstructed_image)
        reconstructed_image[mask] = griddata(
            scaled_points,  # 1. Source points (same as before)
            values,  # 2. Source values (same as before)
            (grid_y[mask], grid_x[mask]),  # 3. Target coords: Only where mask is True
            method='nearest'  # 4. Use nearest to fill gaps/extrapolate
        )

    print(f" [Recon] Output shape: {reconstructed_image.shape}")

    return torch.from_numpy(reconstructed_image)

def reconstruct_hr_image_pytorch(
    sampled_imgs_np: np.ndarray,
    kernels_np: np.ndarray,
    scale_factor: int,
    iterations: int = 160,
    learning_rate: float = 0.01,
    device: str = None
):
    """
    Reconstructs a high-resolution image from multiple low-resolution images using PyTorch.

    This function treats super-resolution as an optimization problem, finding the
    high-resolution image that, when degraded, best matches the observed low-resolution inputs.

    Args:
        sampled_imgs_np: NumPy array of low-resolution images.
                         Shape: [num_images, height, width].
        kernels_np: NumPy array of degradation kernels.
                    Shape: [num_images, kernel_h, kernel_w].
        scale_factor: The integer factor by which to upscale the resolution.
        iterations: The number of optimization iterations to perform.
        learning_rate: The step size for the optimizer.
        device: The device to run the computation on ('cuda', 'cpu', or None).
                If None, it will auto-detect for a CUDA-enabled GPU.

    Returns:
        A NumPy array representing the reconstructed high-resolution image.
        Shape: [height * scale_factor, width * scale_factor].
    """
    # 1. Setup: Device and Data Conversion
    # ---------------------------------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Convert NumPy arrays to PyTorch tensors and move to the selected device
    lr_images = torch.from_numpy(sampled_imgs_np).float().to(device)
    kernels = torch.from_numpy(kernels_np).float().to(device)

    num_images, lr_h, lr_w = lr_images.shape
    hr_h, hr_w = lr_h * scale_factor, lr_w * scale_factor

    # 2. Initialization
    # -----------------
    # Start with an initial guess for the high-resolution image.
    # Upsampling the first low-resolution image using bicubic interpolation is a good start.
    # We add unsqueeze(0) twice to create batch and channel dimensions: [1, 1, H, W].
    hr_image = F.interpolate(
        lr_images[-1].unsqueeze(0).unsqueeze(0),
        size=(hr_h, hr_w),
        mode='bicubic',
        align_corners=False
    ).squeeze() # Squeeze back to [H, W] for now

    # This is the tensor we will optimize. We tell PyTorch to track gradients for it.
    hr_image.requires_grad = True

    # 3. Optimizer and Loss Function
    # ------------------------------
    # Adam is a robust optimizer that often works better than simple gradient descent.
    # We tell it which tensor(s) to optimize.
    optimizer = torch.optim.Adam([hr_image], lr=learning_rate)

    # Mean Squared Error will measure the difference between our simulated LR images
    # and the actual LR images.
    loss_function = torch.nn.MSELoss()

    print("Starting optimization...")
    # 4. Optimization Loop
    # --------------------
    for i in range(iterations):
        # Prepare for a new gradient calculation
        optimizer.zero_grad()

        total_loss = 0

        # Add batch and channel dims for convolution: [H, W] -> [1, 1, H, W]
        hr_image_conv_ready = hr_image.unsqueeze(0).unsqueeze(0)

        # Loop through each low-resolution image and its corresponding kernel
        for j in range(num_images):
            # The kernel needs to be in shape [out_channels, in_channels, kH, kW]
            kernel = kernels[j].unsqueeze(0).unsqueeze(0)

            # --- Forward Pass: Simulate the Degradation ---
            # a. Blur the HR image with the kernel
            blurred_hr = F.conv2d(
                hr_image_conv_ready,
                kernel,
                padding='same'
            )

            # b. Downsample the blurred image (sub-sampling)
            simulated_lr = blurred_hr[:, :, ::scale_factor, ::scale_factor]

            # --- Calculate Loss ---
            # Compare the result with the ground truth low-resolution image
            loss = loss_function(simulated_lr.squeeze(), lr_images[j])
            total_loss += loss

        # --- Backward Pass and Optimization Step ---
        # PyTorch automatically calculates the gradient of the loss
        # with respect to our `hr_image`
        total_loss.backward()

        # The optimizer takes a step in the direction that minimizes the loss
        optimizer.step()

        if (i + 1) % 20 == 0:
            print(f"Iteration [{i+1}/{iterations}], Loss: {total_loss.item():.6f}")

    print("Optimization finished.")
    # 5. Final Result
    # ---------------
    # Detach the tensor from the computation graph, move it to CPU, and convert to NumPy
    reconstructed_image_np = hr_image.detach().cpu().numpy()

    return reconstructed_image_np