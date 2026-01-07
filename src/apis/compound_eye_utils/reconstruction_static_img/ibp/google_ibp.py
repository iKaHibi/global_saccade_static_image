import torch
import torch.nn.functional as F

def list_tensor2device(list_tensor, device):
    res = []
    for tsr in list_tensor:
        res.append(tsr.to(device))
    return res

def iterative_back_projection(
    lr_imgs: list[torch.Tensor],
    kernels: list[torch.Tensor],
    initial_hr_img: torch.Tensor,
    num_iterations: int,
    sf: int,
    alpha: float = 0.1,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Performs Multi-Frame Super-Resolution using the Iterative Back-Projection algorithm.

    Args:
        lr_imgs (list[torch.Tensor]): A list of low-resolution 2D image tensors.
        kernels (list[torch.Tensor]): A list of 2D convolution kernels (PSFs). 
                                     These should already incorporate the sub-pixel shifts.
        initial_hr_img (torch.Tensor): An initial guess for the high-resolution 2D image.
        num_iterations (int): The number of refinement iterations to perform.
        sf (int): The integer scaling factor between LR and HR images.
        alpha (float): A relaxation parameter (learning rate) to control the update step size.

    Returns:
        torch.Tensor: The refined high-resolution image.
    """
    # load all on to device
    lr_imgs = list_tensor2device(lr_imgs[:1], device)  # use the basic image for high density
    kernels = list_tensor2device(kernels[:1], device)
    initial_hr_img = initial_hr_img.to(device)

    # Start with the initial HR guess
    hr_estimate = initial_hr_img.clone()
    # print(f"Starting IBP for {num_iterations} iterations...")

    for i in range(num_iterations):
        # This tensor will accumulate the updates from all 10 frames for the current iteration
        total_back_projection = torch.zeros_like(hr_estimate)

        # Process each LR frame
        for j in range(len(lr_imgs)):
            lr_img = lr_imgs[j]
            kernel = kernels[j]

            # --- Step 1: Forward Projection (Simulate the LR image) ---
            # To use conv2d, we need to add batch and channel dimensions: (B, C, H, W)
            hr_estimate_bchw = hr_estimate.unsqueeze(0).unsqueeze(0)
            kernel_bchw = kernel.unsqueeze(0).unsqueeze(0)

            # Calculate padding to ensure the output of convolution has the same size as the input
            pad_h = (kernel.shape[0] - 1) // 2
            pad_w = (kernel.shape[1] - 1) // 2

            # Apply the blur and shift using convolution
            blurred_hr = F.conv2d(hr_estimate_bchw, kernel_bchw, padding=(pad_h, pad_w))

            # Downsample the blurred HR image to get the simulated LR image
            simulated_lr = blurred_hr[0, 0, ::sf, ::sf]

            # --- Step 2: Calculate the Residual ---
            # This is the difference between the actual observation and our simulation
            residual = lr_img - simulated_lr

            # --- Step 3: Back-Project the Residual ---
            # Upsample the residual by placing it into a zero-filled HR grid
            upsampled_residual = torch.zeros_like(hr_estimate)
            upsampled_residual[::sf, ::sf] = residual
            upsampled_residual_bchw = upsampled_residual.unsqueeze(0).unsqueeze(0)

            # The back-projection kernel is the transpose of the forward kernel.
            # For convolution, this is equivalent to flipping the kernel.
            back_projection_kernel = torch.flip(kernel, dims=[0, 1])
            bp_kernel_bchw = back_projection_kernel.unsqueeze(0).unsqueeze(0)

            # Convolve the upsampled residual with the back-projection kernel
            back_projected_frame = F.conv2d(upsampled_residual_bchw, bp_kernel_bchw, padding=(pad_h, pad_w))

            # Accumulate the result for this frame
            total_back_projection += back_projected_frame.squeeze(0).squeeze(0)
        
        # --- Step 4: Update the HR Estimate ---
        # Add the averaged sum of all back-projected residuals to the current estimate
        # The alpha parameter controls the magnitude of the update
        hr_estimate += alpha * (total_back_projection / len(lr_imgs))
        
        # print(f"  ... Iteration {i + 1}/{num_iterations} complete.")

    # print("IBP finished.")
    return hr_estimate