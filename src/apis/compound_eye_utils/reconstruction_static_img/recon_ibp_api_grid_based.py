import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def debug_coordinate_orientation(pos_rcd):
    """
    Visualizes the coordinate grid.
    If 'pos_rcd' is correct (X, Y):
      - Red (X) should increase from Left -> Right.
      - Green (Y) should increase from Top -> Bottom.
    """
    # Take the first frame's coordinates: (H, W, 2)
    coords = pos_rcd[0]

    # Normalize to 0-1 for visualization
    coords_vis = coords.copy()
    coords_vis[..., 0] = coords_vis[..., 0] / coords_vis[..., 0].max()  # Red channel (should be X)
    coords_vis[..., 1] = coords_vis[..., 1] / coords_vis[..., 1].max()  # Green channel (should be Y)

    # Create an RGB image (Blue is 0)
    debug_img = np.zeros((coords.shape[0], coords.shape[1], 3))
    debug_img[..., 0] = coords_vis[..., 0]  # R = X
    debug_img[..., 1] = coords_vis[..., 1]  # G = Y

    plt.figure(figsize=(6, 6))
    plt.imshow(debug_img)
    plt.title("Coordinate Check\nRed should grow Left->Right\nGreen should grow Top->Bottom")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.show()

    return 0


def prepare_data_for_autograd_ibp(lr_images, pos_rcd, kernels_arr,
                                  original_h, original_w,
                                  recon_size_h, recon_size_w):
    """
    Prepares data, handling the scale difference between the 'Original' physics domain
    and the 'Reconstruction' grid domain.
    """

    # 1. Convert Arrays to Tensor
    lr_tensors = torch.from_numpy(lr_images).float()  # (N, H_lr, W_lr)
    kernel_tensors = torch.from_numpy(kernels_arr).float()  # (N, K_orig, K_orig)

    # 2. Normalize Coordinates (Handling the (x, y) format)
    # Your pos_rcd is (N, H_lr, W_lr, 2) where last dim is (x, y).
    # We normalize x by Width, y by Height.
    grid_norm = np.copy(pos_rcd)

    # Normalize X to [-1, 1] using Original Width
    # (x / (W-1)) * 2 - 1
    grid_norm[..., 0] = (grid_norm[..., 0] / (original_w - 1)) * 2 - 1

    # Normalize Y to [-1, 1] using Original Height
    grid_norm[..., 1] = (grid_norm[..., 1] / (original_h - 1)) * 2 - 1

    # grid_norm[..., 0] = (pos_rcd[..., 1] / (original_w - 1)) * 2 - 1  # X gets data from dim 1
    # grid_norm[..., 1] = (pos_rcd[..., 0] / (original_h - 1)) * 2 - 1  # Y gets data from dim 0

    grid_tensors = torch.from_numpy(grid_norm).float()  # (N, H_lr, W_lr, 2)

    # 3. Handle Kernels (Physics consistency)
    # We must resize the kernel from "Original Scale" to "Reconstruction Scale".
    # Scale Factor = Recon Size / Original Size
    scale_factor_h = recon_size_h / original_h / 0.9
    scale_factor_w = recon_size_w / original_w / 0.9  # the central crop effect

    # Usually we want a single scale factor, assuming square pixels
    scale = (scale_factor_h + scale_factor_w) / 2.0

    kernel_tensors = kernel_tensors.unsqueeze(1)  # Add channel dim: (N, 1, K, K)

    # Downsample the kernel to match the reconstruction resolution
    kernel_recon = F.interpolate(
        kernel_tensors,
        scale_factor=scale,
        mode='bilinear',
        align_corners=False
    )

    # Important: Re-normalize kernel sum to 1 after interpolation to preserve energy
    # Sum over height/width (last 2 dims)
    k_sums = kernel_recon.sum(dim=(2, 3), keepdim=True)
    kernel_recon = kernel_recon / (k_sums + 1e-9)

    return lr_tensors, grid_tensors, kernel_recon


def physics_forward_projection(hr_estimate, grids, kernels):
    """
    hr_estimate: (1, 1, H_recon, W_recon)
    grids: (N, H_lr, W_lr, 2) Normalized coordinates in [-1, 1]
    kernels: (N, 1, K_recon, K_recon) Resized kernels
    """
    num_frames = grids.shape[0]
    simulated_lrs = []

    for i in range(num_frames):
        # A. Blur (Convolution on the Reconstruction Grid)
        k = kernels[i]

        # Calculate padding to keep size same
        # Standard padding = (kernel_size - 1) / 2
        pad_h = k.shape[1] // 2
        pad_w = k.shape[2] // 2

        # Convolve
        blurred_hr = F.conv2d(hr_estimate, k.unsqueeze(0), padding=(pad_h, pad_w))

        # B. Sample (at exact float coordinates)
        # We sample from the blurred HR image.
        # Since grids are normalized to [-1, 1], grid_sample finds the correct
        # relative position regardless of the array size difference (1080 vs 108).
        grid = grids[i].unsqueeze(0)  # (1, H_lr, W_lr, 2)

        sample = F.grid_sample(
            blurred_hr,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )

        simulated_lrs.append(sample)

    return torch.cat(simulated_lrs, dim=0)


def total_variation_loss(img, weight):
    diff_h = torch.abs(img[..., 1:, :] - img[..., :-1, :])
    diff_w = torch.abs(img[..., :, 1:] - img[..., :, :-1])
    return weight * (diff_h.mean() + diff_w.mean())

def solve_ibp_autograd(lr_tensors, grid_tensors, kernel_recon, initial_hr_guess, iterations=64, lr=0.5):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    real_lrs = lr_tensors.unsqueeze(1).to(device)
    grids = grid_tensors.to(device)
    kernels = kernel_recon.to(device)
    hr_estimate = initial_hr_guess.clone().detach().to(device).requires_grad_(True)

    optimizer = torch.optim.SGD([hr_estimate], lr=lr)
    # Using a scheduler helps converge finely at the end
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # print(f" [IBP] Starting Autograd reconstruction...")

    for i in range(iterations):
        optimizer.zero_grad()

        simulated_lrs = physics_forward_projection(hr_estimate, grids, kernels)

        loss = F.mse_loss(simulated_lrs, real_lrs, reduction='sum') + total_variation_loss(hr_estimate, weight=0.1)

        loss.backward()
        optimizer.step()
        scheduler.step()

        # if i % 10 == 0:
        #     print(f" [IBP] Iter {i}: MSE Loss {loss.item():.4f}")

    return hr_estimate.detach()