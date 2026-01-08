"""
This file contains function to calculate simple metrics including: (ALL CHECKED MANUALLY, NY UNIT TEST)
- mean_frequency_spectrum
- frequency spectrum std
- frequency_slope

The API defines as:
input:
    a batch of 1 channel grey value images, tensor (B, S, S)
    device to calculate, string, 'cpu'/'cuda'
output:
    metrics, tensor (B, )
"""

import torch
import numpy as np

def frequency_mean_api_torch(images, freq_threshold=8, device='cuda'):
    """
    Calculate the following features of grayscale images using PyTorch on GPU:
        - frequency magnitude mean value
        - frequency magnitude standard deviation
        - frequency component decay slope

    Parameters:
    images (numpy array or torch tensor): Input images. Can be 2D (single image) or 3D (batch of images).
    device (str): Device to use, e.g., 'cuda' or 'cpu'.

    Returns:
        freq_mag_mean: torch.Tensor, (B, ),  magnitude mean for each image
        freq_mag_std: torch.Tensor, (B, ),  magnitude std for each image.
        torch.tensor(slopes): torch.Tensor, (B, ),  Slopes for each image.
    """
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images).float()
    elif not isinstance(images, torch.Tensor):
        raise ValueError("Images must be a numpy array or torch tensor")

    if images.ndim == 2:
        images = images.unsqueeze(0)  # Add batch dimension
    elif images.ndim != 3:
        raise ValueError("Images tensor musc be in shape [B, H, W]")

    B, M, N = images.shape
    if M < 10 or N < 10:
        raise ValueError("Images are too small to compute slope reliably")

    images = images.to(device=device)

    # Compute FFT
    fft = torch.fft.fft2(images, dim=(-2, -1))
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
    amplitude = torch.abs(fft_shifted)

    # compute the simple frequency features
    # Create distance grid
    i = torch.arange(M, device=device)
    j = torch.arange(N, device=device)
    I, J = torch.meshgrid(i, j, indexing='ij')
    center_i = M // 2
    center_j = N // 2
    distance = torch.sqrt((I.float() - center_i) ** 2 + (J.float() - center_j) ** 2)
    mask = distance > freq_threshold

    # Select amplitudes
    amplitude_flat = amplitude.reshape(B, -1)
    mask_flat = mask.view(-1)
    if mask_flat.sum() > 0:
        selected_amplitudes = amplitude_flat[:, mask_flat]
        freq_mag_mean = selected_amplitudes.mean(dim=1)
    else:
        freq_mag_mean = torch.zeros(B, device=device)

    return freq_mag_mean


def frequency_std_api_torch(images, freq_threshold=8, device='cuda'):
    """
    Calculate the following features of grayscale images using PyTorch on GPU:
        - frequency magnitude mean value
        - frequency magnitude standard deviation
        - frequency component decay slope

    Parameters:
    images (numpy array or torch tensor): Input images. Can be 2D (single image) or 3D (batch of images).
    device (str): Device to use, e.g., 'cuda' or 'cpu'.

    Returns:
        freq_mag_mean: torch.Tensor, (B, ),  magnitude mean for each image
        freq_mag_std: torch.Tensor, (B, ),  magnitude std for each image.
        torch.tensor(slopes): torch.Tensor, (B, ),  Slopes for each image.
    """
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images).float()
    elif not isinstance(images, torch.Tensor):
        raise ValueError("Images must be a numpy array or torch tensor")

    if images.ndim == 2:
        images = images.unsqueeze(0)  # Add batch dimension
    elif images.ndim != 3:
        raise ValueError("Images tensor musc be in shape [B, H, W]")

    B, M, N = images.shape
    if M < 10 or N < 10:
        raise ValueError("Images are too small to compute slope reliably")

    images = images.to(device=device)

    # Compute FFT
    fft = torch.fft.fft2(images, dim=(-2, -1))
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
    amplitude = torch.abs(fft_shifted)

    # compute the simple frequency features
    # Create distance grid
    i = torch.arange(M, device=device)
    j = torch.arange(N, device=device)
    I, J = torch.meshgrid(i, j, indexing='ij')
    center_i = M // 2
    center_j = N // 2
    distance = torch.sqrt((I.float() - center_i) ** 2 + (J.float() - center_j) ** 2)
    mask = distance > freq_threshold

    # Select amplitudes
    amplitude_flat = amplitude.reshape(B, -1)
    mask_flat = mask.view(-1)
    if mask_flat.sum() > 0:
        selected_amplitudes = amplitude_flat[:, mask_flat]
        freq_mag_std = selected_amplitudes.std(dim=1)
    else:
        freq_mag_std = torch.zeros(B, device=device)

    return freq_mag_std


def frequency_slope_api_torch(images, freq_threshold=8, device='cuda'):
    """
    Calculate the following features of grayscale images using PyTorch on GPU:
        - frequency magnitude mean value
        - frequency magnitude standard deviation
        - frequency component decay slope

    Parameters:
    images (numpy array or torch tensor): Input images. Can be 2D (single image) or 3D (batch of images).
    device (str): Device to use, e.g., 'cuda' or 'cpu'.

    Returns:
        freq_mag_mean: torch.Tensor, (B, ),  magnitude mean for each image
        freq_mag_std: torch.Tensor, (B, ),  magnitude std for each image.
        torch.tensor(slopes): torch.Tensor, (B, ),  Slopes for each image.
    """
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images).float()
    elif not isinstance(images, torch.Tensor):
        raise ValueError("Images must be a numpy array or torch tensor")

    if images.ndim == 2:
        images = images.unsqueeze(0)  # Add batch dimension
    elif images.ndim != 3:
        raise ValueError("Images tensor musc be in shape [B, H, W]")

    B, M, N = images.shape
    if M < 10 or N < 10:
        raise ValueError("Images are too small to compute slope reliably")

    images = images.to(device=device)

    # Compute FFT
    fft = torch.fft.fft2(images, dim=(-2, -1))
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
    amplitude = torch.abs(fft_shifted)

    # compute the simple frequency features
    # Create distance grid
    i = torch.arange(M, device=device)
    j = torch.arange(N, device=device)
    I, J = torch.meshgrid(i, j, indexing='ij')
    center_i = M // 2
    center_j = N // 2
    distance = torch.sqrt((I.float() - center_i) ** 2 + (J.float() - center_j) ** 2)
    mask = distance > freq_threshold

    # Select amplitudes
    amplitude_flat = amplitude.reshape(B, -1)
    mask_flat = mask.view(-1)
    if mask_flat.sum() > 0:
        selected_amplitudes = amplitude_flat[:, mask_flat]
        freq_mag_mean = selected_amplitudes.mean(dim=1)
        freq_mag_std = selected_amplitudes.std(dim=1)
    else:
        freq_mag_mean = torch.zeros(B, device=device)
        freq_mag_std = torch.zeros(B, device=device)

    # Compute radial distances
    center = (M // 2, N // 2)
    rows = torch.arange(M, device=device) - center[0]
    cols = torch.arange(N, device=device) - center[1]
    rows, cols = torch.meshgrid(rows, cols, indexing='ij')
    r = torch.sqrt(rows**2 + cols**2)
    r_flat = r.flatten()
    bin_indices = r_flat.round().long()
    max_bin = bin_indices.max().item()

    # Prepare sums and counts
    sums = torch.zeros((B, max_bin + 1), device=device)
    counts = torch.zeros((B, max_bin + 1), device=device)

    amplitude_flat = amplitude.view(B, -1)
    for b in range(B):
        sums[b].scatter_add_(0, bin_indices, amplitude_flat[b])
        counts[b].scatter_add_(0, bin_indices, torch.ones_like(amplitude_flat[b], device=device))

    means = sums / counts

    # Compute slopes
    slopes = []
    for b in range(B):
        if max_bin > 128:
            valid_bin = 128
        else:
            valid_bin = max_bin + 1
        rp_valid = means[b, 1:valid_bin]
        r_valid = torch.arange(1, valid_bin, device=device).float()
        mask = ~torch.isnan(rp_valid) & (rp_valid > 0)
        r_valid = r_valid[mask]
        rp_valid = rp_valid[mask]
        if len(r_valid) < 2:
            slopes.append(np.nan)
            continue
        log_r = torch.log(r_valid)
        log_rp = torch.log(rp_valid)
        N = len(log_r)
        sum_x = log_r.sum()
        sum_y = log_rp.sum()
        sum_xy = (log_r * log_rp).sum()
        sum_x2 = (log_r ** 2).sum()
        numerator = N * sum_xy - sum_x * sum_y
        denominator = N * sum_x2 - sum_x ** 2
        if denominator == 0:
            slopes.append(np.nan)
        else:
            slope = numerator / denominator
            slopes.append(slope.item())

    return torch.tensor(slopes)

