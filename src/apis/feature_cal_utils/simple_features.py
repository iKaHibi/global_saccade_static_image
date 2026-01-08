"""
This file contains function to calculate simple metrics including:
- luminance mean
- luminance variance
- contrast
- entropy
- simple sharpness
- coarseness
- roughness
- line-likeness (not checked)
- regularity (not checked)

The API defines as:
input:
    a batch of 1 channel grey value images, tensor (B, S, S)
    device to calculate, string, 'cpu'/'cuda'
output:
    metrics, tensor (B, )
"""

import torch
import torch.nn.functional as F
import numpy as np


def coarseness_api_pytorch(images, device='cpu', kmax=5):
    """
    Compute Tamura's coarseness feature for a batch of grayscale images.

    Parameters:
    - images (torch.Tensor): Batch of grayscale images, shape (B, S, S).
    - device (str): Device to perform computations ('cpu' or 'cuda').
    - kmax (int): Maximum scale for window sizes (default=5).

    Returns:
    - torch.Tensor: Coarseness values for each image, shape (B,).
    """
    images = images.to(device)
    B, S, S = images.shape
    kmax = min(kmax, int(np.log2(S)))
    horizon_all = []
    vertical_all = []

    for k in range(kmax):
        window = 2 ** k
        kernel_size = 2 * window
        p = kernel_size // 2
        # Compute averages with padding to maintain size
        average_k = F.avg_pool2d(images.unsqueeze(1), kernel_size=kernel_size, stride=1, padding=p)
        # Crop or pad to ensure size (B, 1, S, S)
        if average_k.size(2) > S:
            average_k = average_k[:, :, :S, :S]
        elif average_k.size(2) < S:
            pad_size = S - average_k.size(2)
            average_k = F.pad(average_k, (0, pad_size, 0, pad_size), mode='constant', value=0)

        # Initialize difference tensors
        horizon_k = torch.zeros((B, S, S), device=device)
        vertical_k = torch.zeros((B, S, S), device=device)
        start = window
        end = S - window

        # Compute differences for valid positions
        if start < end:
            horizon_k[:, start:end, :] = (
                    average_k[:, 0, 2 * window:S, :] - average_k[:, 0, 0:S - 2 * window, :]
            )
            vertical_k[:, :, start:end] = (
                    average_k[:, 0, :, 2 * window:S] - average_k[:, 0, :, 0:S - 2 * window]
            )

        # Normalize differences
        norm_factor = (2 ** (k + 1)) ** 2
        horizon_k = horizon_k / norm_factor
        vertical_k = vertical_k / norm_factor

        horizon_all.append(horizon_k)
        vertical_all.append(vertical_k)

    # Stack differences across scales
    horizon_all = torch.stack(horizon_all, dim=1)  # (B, kmax, S, S)
    vertical_all = torch.stack(vertical_all, dim=1)  # (B, kmax, S, S)

    # Compute maximum differences and select best scale
    h_max = torch.max(torch.abs(horizon_all), dim=1)[0]  # (B, S, S)
    v_max = torch.max(torch.abs(vertical_all), dim=1)[0]  # (B, S, S)
    mask = h_max > v_max  # (B, S, S)
    indices_h = torch.argmax(torch.abs(horizon_all), dim=1)  # (B, S, S)
    indices_v = torch.argmax(torch.abs(vertical_all), dim=1)  # (B, S, S)
    Sbest_index = torch.where(mask, indices_h, indices_v)  # (B, S, S)
    Sbest = 2.0 ** Sbest_index.float()  # (B, S, S)

    # Compute mean coarseness per image
    coarseness = Sbest.mean(dim=[1, 2])  # (B,)
    return coarseness

def contrast_api_pytorch(images, device='cpu'):
    images = images.to(device)
    mean = images.mean(dim=[1,2], keepdim=True)
    std = images.std(dim=[1,2], keepdim=True)
    kurt = ((images - mean) ** 4).mean(dim=[1,2], keepdim=True) / (std ** 4 + 1e-8)
    contrast = std.squeeze() / (kurt.squeeze() ** 0.25)
    return contrast

def simple_sharpness_api_pytorch(images, device='cpu'):
    images = images.to(device)
    kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    laplacian = F.conv2d(images.unsqueeze(1), kernel, padding=1)
    sharpness = laplacian.var(dim=[2,3]).squeeze()
    return sharpness

def roughness_api_pytorch(images, device='cpu'):
    coarseness = coarseness_api_pytorch(images, device)
    contrast = contrast_api_pytorch(images, device)
    return coarseness + contrast

def calculate_luminance_mean_pytorch(images, device='cpu'):
    images = images.to(device)
    return images.mean(dim=[1,2])

def calculate_luminance_variance_pytorch(images, device='cpu'):
    images = images.to(device)
    return images.var(dim=[1,2])

def calculate_entropy_pytorch(images, device='cpu', bins=256):
    images = images.to(device)
    B = images.size(0)
    entropies = torch.zeros(B, device=device)
    for b in range(B):
        hist = torch.histc(images[b], bins=bins, min=0, max=1)
        p = hist / hist.sum()
        p = p[p > 0]
        entropies[b] = - (p * torch.log2(p)).sum()
    return entropies

def line_likeness_api_pytorch(images, device='cpu', threshold=None):
    images = images.to(device)
    B, S, S = images.shape
    images = images.unsqueeze(1)  # (B, 1, S, S)

    # Define Sobel filters
    sobel_h = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    sobel_v = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    # Compute gradients
    dx = F.conv2d(images, sobel_h, padding=1).squeeze(1)  # (B, S, S)
    dy = F.conv2d(images, sobel_v, padding=1).squeeze(1)  # (B, S, S)

    # Compute magnitude and theta
    magnitude = torch.sqrt(dx**2 + dy**2)
    theta = (torch.atan2(dy, dx) * (180 / torch.pi) + 360) % 180  # (B, S, S)

    # Compute threshold if not provided
    if threshold is None:
        threshold = magnitude.mean(dim=[1,2])  # (B,)
    else:
        if isinstance(threshold, (int, float)):
            threshold = torch.full((B,), threshold, device=device)
        elif isinstance(threshold, torch.Tensor):
            threshold = threshold.to(device)
        else:
            raise ValueError("threshold must be a number or a tensor")

    # Create mask
    mask = magnitude > threshold[:, None, None]  # (B, S, S)

    # Get indices where mask is True
    indices = torch.nonzero(mask, as_tuple=False)  # (N, 3), columns: b, r, c
    if indices.numel() == 0:
        return torch.zeros(B, device=device)

    b_idx = indices[:, 0]
    r_idx = indices[:, 1]
    c_idx = indices[:, 2]
    theta_selected = theta[b_idx, r_idx, c_idx]

    # Compute bins
    bins = torch.zeros_like(theta_selected, dtype=torch.long)
    bins[(theta_selected < 22.5) | (theta_selected >= 157.5)] = 0
    bins[(theta_selected >= 22.5) & (theta_selected < 67.5)] = 1
    bins[(theta_selected >= 67.5) & (theta_selected < 112.5)] = 2
    bins[(theta_selected >= 112.5) & (theta_selected < 157.5)] = 3

    # Define offsets
    di = torch.tensor([0, -1, -1, -1], device=device)
    dj = torch.tensor([1, 1, 0, -1], device=device)

    # Compute neighbor coordinates
    nr = r_idx + di[bins]
    nc = c_idx + dj[bins]

    # Check valid neighbors
    valid = (nr >= 0) & (nr < S) & (nc >= 0) & (nc < S)
    valid_indices = torch.where(valid)[0]
    if valid_indices.numel() == 0:
        return torch.zeros(B, device=device)

    b_valid = b_idx[valid_indices]
    r_valid = nr[valid_indices]
    c_valid = nc[valid_indices]
    theta_neighbor = theta[b_valid, r_valid, c_valid]
    theta_pixel = theta_selected[valid_indices]

    # Compute delta_theta
    abs_diff = torch.abs(theta_pixel - theta_neighbor)
    delta_theta = torch.min(abs_diff, 180 - abs_diff)

    # Compute cos(2 * delta_theta * pi / 180)
    cos_values = torch.cos(2 * delta_theta * torch.pi / 180)

    # Sum cos_values and counts per batch
    cos_sums = torch.zeros(B, device=device)
    counts = torch.zeros(B, device=device)
    cos_sums.scatter_add_(0, b_valid, cos_values)
    counts.scatter_add_(0, b_valid, torch.ones_like(cos_values))

    # Compute mean cos per image
    mean_cos = cos_sums / (counts + 1e-8)
    mean_cos = torch.where(counts > 0, mean_cos, torch.zeros_like(mean_cos))

    return mean_cos


def spatial_information_pytorch(images, device='cpu'):
    """
    Compute Spatial Information (SI) for a batch of grayscale images.
    SI is defined as the standard deviation of the Sobel-filtered image.

    Parameters:
    - images (torch.Tensor): Batch of grayscale images, shape (B, S, S).
    - device (str): Device for computation ('cpu' or 'cuda').

    Returns:
    - torch.Tensor: SI values for each image, shape (B,).
    """
    images = images.to(device).unsqueeze(1)  # Shape: (B, 1, S, S)
    
    # Define Sobel kernels for horizontal and vertical gradients
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=torch.float32, device=device).view(1, 1, 3, 3)

    # Apply filters using 2D convolution
    grad_x = F.conv2d(images, sobel_x, padding=1)
    grad_y = F.conv2d(images, sobel_y, padding=1)

    # Compute gradient magnitude: sqrt(grad_x^2 + grad_y^2)
    magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

    # SI is the standard deviation of the magnitude spatial plane
    # Reshape to (B, -1) to calculate std per image
    si = magnitude.view(images.size(0), -1).std(dim=1)
    
    return si

def directionality_api_pytorch(images, device='cpu'):
    images = images.to(device)
    B, S, S = images.shape
    images = images.unsqueeze(1)  # (B, 1, S, S)

    # Define convolution kernels
    convH = torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    convV = torch.tensor([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    # Compute deltaH and deltaV
    deltaH = F.conv2d(images, convH, padding=1).squeeze(1)  # (B, S, S)
    deltaV = F.conv2d(images, convV, padding=1).squeeze(1)  # (B, S, S)

    # Compute deltaG and theta
    deltaG = (torch.abs(deltaH) + torch.abs(deltaV)) / 2.0  # (B, S, S)
    theta = (torch.atan2(deltaV, deltaH) + torch.pi / 2) * (180 / torch.pi) % 360  # (B, S, S)

    n = 16
    t = 12
    fdir = torch.zeros(B, device=device)

    for b in range(B):
        mask_b = deltaG[b] > t
        if mask_b.sum() == 0:
            continue
        theta_b = theta[b][mask_b]
        bins_b = torch.floor((n * theta_b) / 360).long() % n
        hd_b = torch.bincount(bins_b, minlength=n)
        max_pos = torch.argmax(hd_b)
        i = torch.arange(n, device=device)
        fdir[b] = ((i - max_pos)**2 * hd_b).sum()

    return fdir


def regularity_api_pytorch(images, device='cpu', num_subimages=4):
    images = images.to(device)
    B, S, S = images.shape
    assert S % num_subimages == 0, "S must be divisible by num_subimages"
    sub_S = S // num_subimages

    # Split into subimages
    sub_images = images.view(B, num_subimages, sub_S, num_subimages, sub_S)
    sub_images = sub_images.permute(0,1,3,2,4).contiguous().view(B*num_subimages**2, sub_S, sub_S)

    # Compute features for subimages
    crs = coarseness_api_pytorch(sub_images, device=device).view(B, num_subimages**2)
    con = contrast_api_pytorch(sub_images, device=device).view(B, num_subimages**2)
    dir_ = directionality_api_pytorch(sub_images, device=device).view(B, num_subimages**2)
    lin = line_likeness_api_pytorch(sub_images, device=device).view(B, num_subimages**2)

    # Compute std per feature across subimages
    std_crs = crs.std(dim=1)
    std_con = con.std(dim=1)
    std_dir = dir_.std(dim=1)
    std_lin = lin.std(dim=1)

    sum_std = std_crs + std_con + std_dir + std_lin
    r = 0.25
    regularity = 1 - r * sum_std
    regularity = torch.clamp(regularity, 0.0, 1.0)

    return regularity


def skewness_api_pytorch(images, device='cpu'):
    """
    Calculate the skewness of the pixel distribution (Asymmetry).
    """
    images = images.to(device)
    mean = images.mean(dim=[1, 2], keepdim=True)
    std = images.std(dim=[1, 2], keepdim=True)
    # Third central moment
    m3 = ((images - mean) ** 3).mean(dim=[1, 2], keepdim=True)
    skew = m3 / (std ** 3 + 1e-8)
    return skew.squeeze()

def kurtosis_api_pytorch(images, device='cpu'):
    """
    Calculate the kurtosis of the pixel distribution (Tail heaviness).
    Fisher kurtosis (normal distribution = 0.0) is often used, but Pearson 
    (normal = 3.0) is standard in some libraries. Here we return Pearson.
    """
    images = images.to(device)
    mean = images.mean(dim=[1, 2], keepdim=True)
    std = images.std(dim=[1, 2], keepdim=True)
    # Fourth central moment
    m4 = ((images - mean) ** 4).mean(dim=[1, 2], keepdim=True)
    kurt = m4 / (std ** 4 + 1e-8)
    return kurt.squeeze()

def spectral_slope_api_pytorch(images, device='cpu'):
    """
    Calculate the spectral slope (alpha) of the images.
    Natural images usually have a power spectrum P(f) ~ 1/f^alpha.
    We estimate alpha by fitting a line to log(Power) vs log(Frequency).
    """
    images = images.to(device)
    B, H, W = images.shape
    
    # Compute FFT
    fft = torch.fft.fft2(images)
    fft = torch.fft.fftshift(fft, dim=(-2, -1))
    power = (torch.abs(fft) ** 2) / (H * W)
    
    # Create frequency grid
    y = torch.arange(H, device=device) - H // 2
    x = torch.arange(W, device=device) - W // 2
    Y, X = torch.meshgrid(y, x, indexing='ij')
    freq = torch.sqrt(X**2 + Y**2)
    
    # Flatten and filter out DC component (freq=0) and corners
    freq = freq.flatten()
    
    # We fit the slope for each image in the batch
    slopes = torch.zeros(B, device=device)
    
    # Mask for valid frequencies (avoid DC and very high freq artifacts)
    # We typically fit in the middle frequency range
    mask = (freq > 0) & (freq < min(H, W) / 2)
    valid_freq = freq[mask]
    log_freq = torch.log(valid_freq)
    
    # Pre-calculate variance for linear regression denominator
    mean_log_freq = log_freq.mean()
    denom = ((log_freq - mean_log_freq) ** 2).sum()
    
    for b in range(B):
        p = power[b].flatten()
        valid_power = p[mask]
        log_power = torch.log(valid_power + 1e-12) # Avoid log(0)
        
        # Simple Linear Regression: Slope = Cov(x,y) / Var(x)
        # We want alpha where log(P) = -alpha * log(f) + c
        # So we expect a negative slope
        
        mean_log_power = log_power.mean()
        num = ((log_freq - mean_log_freq) * (log_power - mean_log_power)).sum()
        
        # The slope is usually negative (-alpha), so we return -slope as alpha
        slope = num / (denom + 1e-8)
        slopes[b] = -slope 
        
    return slopes