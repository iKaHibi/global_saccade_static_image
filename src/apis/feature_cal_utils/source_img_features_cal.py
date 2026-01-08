import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm  # For progress bar


# ==========================================
# 1. Metric Implementations
# ==========================================

def coarseness_api_pytorch(images, device='cpu', kmax=5):
    images = images.to(device)
    B, H, W = images.shape
    # Ensure square or handle non-square by taking min dim for log2
    kmax = min(kmax, int(np.log2(min(H, W))))
    horizon_all = []
    vertical_all = []

    for k in range(kmax):
        window = 2 ** k
        kernel_size = 2 * window
        p = kernel_size // 2

        # Average pooling
        average_k = F.avg_pool2d(images.unsqueeze(1), kernel_size=kernel_size, stride=1, padding=p)

        # Trim to original size if padding expanded it
        if average_k.size(2) > H: average_k = average_k[:, :, :H, :]
        if average_k.size(3) > W: average_k = average_k[:, :, :, :W]

        # Calculate differences
        horizon_k = torch.zeros_like(images)
        vertical_k = torch.zeros_like(images)

        # Valid region for differences
        # Note: This simple slicing assumes padding handled boundaries,
        # but strictest Tamura impl crops invalid edges.
        # Here we stick to your implementation logic but safe-guard indices.
        h_start, h_end = window, H - window
        w_start, w_end = window, W - window

        if h_start < h_end and w_start < w_end:
            # Horizontal difference (shift vertical axis)
            # Check indices carefully.
            # Tamura horizontal diff: difference between left and right windows.
            # Your code: average_k[:, 0, 2*window:S, :] - ... (shifting Height?)
            # Usually horizontal diff means checking neighbors left/right.
            # Assuming your implementation intent is correct for "Horizon":

            # Using safe slicing based on your provided logic:
            valid_h = average_k[:, 0, 2 * window:H, :] - average_k[:, 0, 0:H - 2 * window, :]
            valid_v = average_k[:, 0, :, 2 * window:W] - average_k[:, 0, :, 0:W - 2 * window]

            # Place into full size tensor
            horizon_k[:, window:H - window, :] = valid_h
            vertical_k[:, :, window:W - window] = valid_v

        norm_factor = (2 ** (k + 1)) ** 2
        horizon_all.append(horizon_k / norm_factor)
        vertical_all.append(vertical_k / norm_factor)

    horizon_all = torch.stack(horizon_all, dim=1)
    vertical_all = torch.stack(vertical_all, dim=1)

    # Find best scale
    h_max = torch.max(torch.abs(horizon_all), dim=1)[0]
    v_max = torch.max(torch.abs(vertical_all), dim=1)[0]

    mask = h_max > v_max
    indices_h = torch.argmax(torch.abs(horizon_all), dim=1)
    indices_v = torch.argmax(torch.abs(vertical_all), dim=1)
    Sbest_index = torch.where(mask, indices_h, indices_v)
    Sbest = 2.0 ** Sbest_index.float()

    return Sbest.mean(dim=[1, 2])


def contrast_api_pytorch(images, device='cpu'):
    images = images.to(device)
    mean = images.mean(dim=[1, 2], keepdim=True)
    std = images.std(dim=[1, 2], keepdim=True)
    kurt = ((images - mean) ** 4).mean(dim=[1, 2], keepdim=True) / (std ** 4 + 1e-8)
    contrast = std.squeeze() / (kurt.squeeze() ** 0.25)
    return contrast


def simple_sharpness_api_pytorch(images, device='cpu'):
    images = images.to(device)
    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=device).unsqueeze(
        0).unsqueeze(0)
    laplacian = F.conv2d(images.unsqueeze(1), kernel, padding=1)
    sharpness = laplacian.var(dim=[2, 3]).squeeze()
    return sharpness


def roughness_api_pytorch(images, device='cpu'):
    coarseness = coarseness_api_pytorch(images, device)
    contrast = contrast_api_pytorch(images, device)
    return coarseness + contrast


def calculate_luminance_mean_pytorch(images, device='cpu'):
    return images.to(device).mean(dim=[1, 2])



def calculate_entropy_pytorch(images, device='cpu', bins=256):
    images = images.to(device)
    B = images.size(0)
    entropies = torch.zeros(B, device=device)
    for b in range(B):
        # Explicitly specifying range [0, 1] as inputs are normalized
        hist = torch.histc(images[b], bins=bins, min=0, max=1)
        p = hist / hist.sum()
        p = p[p > 0]
        entropies[b] = - (p * torch.log2(p)).sum()
    return entropies


def line_likeness_api_pytorch(images, device='cpu', threshold=None):
    images = images.to(device)
    B, H, W = images.shape
    images_unsqueezed = images.unsqueeze(1)

    sobel_h = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_v = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

    dx = F.conv2d(images_unsqueezed, sobel_h, padding=1).squeeze(1)
    dy = F.conv2d(images_unsqueezed, sobel_v, padding=1).squeeze(1)

    magnitude = torch.sqrt(dx ** 2 + dy ** 2)
    theta = (torch.atan2(dy, dx) * (180 / torch.pi) + 360) % 180

    if threshold is None:
        threshold = magnitude.mean(dim=[1, 2])
    elif isinstance(threshold, (int, float)):
        threshold = torch.full((B,), threshold, device=device)

    mask = magnitude > threshold[:, None, None]

    # Return 0 if no edges detected
    if mask.sum() == 0: return torch.zeros(B, device=device)

    # Simplified Vectorized approach to avoid massive loops
    # Note: The original neighbor logic is complex to fully vectorize without indices.
    # We will use the original index-based logic but ensure it handles empty masks safely.

    indices = torch.nonzero(mask, as_tuple=False)
    b_idx, r_idx, c_idx = indices[:, 0], indices[:, 1], indices[:, 2]
    theta_selected = theta[b_idx, r_idx, c_idx]

    bins = torch.zeros_like(theta_selected, dtype=torch.long)
    bins[(theta_selected >= 22.5) & (theta_selected < 67.5)] = 1
    bins[(theta_selected >= 67.5) & (theta_selected < 112.5)] = 2
    bins[(theta_selected >= 112.5) & (theta_selected < 157.5)] = 3

    di = torch.tensor([0, -1, -1, -1], device=device)
    dj = torch.tensor([1, 1, 0, -1], device=device)

    nr = r_idx + di[bins]
    nc = c_idx + dj[bins]

    valid = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
    valid_indices = torch.where(valid)[0]

    if valid_indices.numel() == 0: return torch.zeros(B, device=device)

    b_valid = b_idx[valid_indices]
    theta_neighbor = theta[b_valid, nr[valid_indices], nc[valid_indices]]
    theta_pixel = theta_selected[valid_indices]

    abs_diff = torch.abs(theta_pixel - theta_neighbor)
    delta_theta = torch.min(abs_diff, 180 - abs_diff)
    cos_values = torch.cos(2 * delta_theta * torch.pi / 180)

    cos_sums = torch.zeros(B, device=device)
    counts = torch.zeros(B, device=device)
    cos_sums.scatter_add_(0, b_valid, cos_values)
    counts.scatter_add_(0, b_valid, torch.ones_like(cos_values))

    return torch.where(counts > 0, cos_sums / counts, torch.zeros_like(cos_sums))


def directionality_api_pytorch(images, device='cpu'):
    images = images.to(device)
    B, _, _ = images.shape
    images_u = images.unsqueeze(1)

    # 1. Detect if images are [0, 1] or [0, 255] to set threshold
    if images.max() <= 1.0:
        t = 0.05  # Approx 12 / 255
    else:
        t = 12.0

    convH = torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    convV = torch.tensor([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

    deltaH = F.conv2d(images_u, convH, padding=1).squeeze(1)
    deltaV = F.conv2d(images_u, convV, padding=1).squeeze(1)

    deltaG = (torch.abs(deltaH) + torch.abs(deltaV)) / 2.0
    theta = (torch.atan2(deltaV, deltaH) + torch.pi / 2) * (180 / torch.pi) % 360

    n = 16
    fdir = torch.zeros(B, device=device)

    for b in range(B):
        mask_b = deltaG[b] > t

        # If the image is completely flat/smooth, directionality is 0
        if mask_b.sum() == 0:
            continue

        theta_b = theta[b][mask_b]
        bins_b = torch.floor((n * theta_b) / 360).long() % n
        hd_b = torch.bincount(bins_b, minlength=n).float()

        # Normalize histogram
        if hd_b.sum() > 0:
            max_pos = torch.argmax(hd_b)
            i = torch.arange(n, device=device)

            # Circular distance to peak
            diff = torch.abs(i - max_pos)
            diff = torch.min(diff, n - diff)

            # Tamura Directionality is 1 - r * sum(...)
            # But standard implementations often return the raw second moment.
            # If you want strictly "Higher = More Directional", you might want to invert this.
            # However, keeping your original logic (Second Moment):
            fdir[b] = ((diff ** 2) * hd_b).sum() / hd_b.sum()

    return fdir


def regularity_api_pytorch(images, device='cpu', num_subimages=4):
    images = images.to(device)
    B, H, W = images.shape

    # Check divisibility
    if H % num_subimages != 0 or W % num_subimages != 0:
        # Return NaN or 0 if dimensions don't match, to avoid crash
        return torch.full((B,), float('nan'), device=device)

    sub_H = H // num_subimages
    sub_W = W // num_subimages

    # Reshape to create sub-images
    # (B, H, W) -> (B, num, sub_H, num, sub_W) -> (B, num, num, sub_H, sub_W)
    sub_images = images.view(B, num_subimages, sub_H, num_subimages, sub_W)
    sub_images = sub_images.permute(0, 1, 3, 2, 4).contiguous()
    # Collapse into batch dimension: (B * num * num, sub_H, sub_W)
    sub_images = sub_images.view(-1, sub_H, sub_W)

    # Compute features on sub-images
    # Use try-except to handle potential shape issues recursively
    try:
        crs = coarseness_api_pytorch(sub_images, device).view(B, -1)
        con = contrast_api_pytorch(sub_images, device).view(B, -1)
        dir_ = directionality_api_pytorch(sub_images, device).view(B, -1)
        lin = line_likeness_api_pytorch(sub_images, device).view(B, -1)
    except Exception as e:
        return torch.full((B,), float('nan'), device=device)

    # Standard deviation across sub-images (normalized by global scale helps, but raw is standard)
    std_crs = crs.std(dim=1)
    std_con = con.std(dim=1)
    std_dir = dir_.std(dim=1)
    std_lin = lin.std(dim=1)

    sum_std = std_crs + std_con + std_dir + std_lin
    r = 0.25  # Sensitivity factor
    regularity = 1 - r * sum_std
    regularity = torch.clamp(regularity, 0.0, 1.0)

    return regularity


def skewness_api_pytorch(images, device='cpu'):
    images = images.to(device)
    mean = images.mean(dim=[1, 2], keepdim=True)
    std = images.std(dim=[1, 2], keepdim=True)
    m3 = ((images - mean) ** 3).mean(dim=[1, 2], keepdim=True)
    skew = m3 / (std ** 3 + 1e-8)
    return skew.squeeze()


def kurtosis_api_pytorch(images, device='cpu'):
    images = images.to(device)
    mean = images.mean(dim=[1, 2], keepdim=True)
    std = images.std(dim=[1, 2], keepdim=True)
    m4 = ((images - mean) ** 4).mean(dim=[1, 2], keepdim=True)
    kurt = m4 / (std ** 4 + 1e-8)
    return kurt.squeeze()


def spectral_slope_api_pytorch(images, freq_threshold=8, device='cuda'):
    """
    Computes Spectral Slope using 1D Rotational Averaging (Classic Method).
    Calculates on Power Spectrum (Amplitude^2).
    """
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images).float()

    if images.ndim == 2:
        images = images.unsqueeze(0)

    images = images.to(device)
    B, H, W = images.shape

    # 1. Compute FFT and Power Spectrum
    fft = torch.fft.fft2(images)
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
    amplitude = torch.abs(fft_shifted)
    power = amplitude ** 2  # <--- CHANGE: Standard Slope uses Power

    # 2. Simple High-Freq Statistics (from your code)
    # Create radial grid
    y = torch.arange(H, device=device) - H // 2
    x = torch.arange(W, device=device) - W // 2
    Y, X = torch.meshgrid(y, x, indexing='ij')
    radius = torch.sqrt(X ** 2 + Y ** 2)

    # Mean/Std of high freq magnitude
    mask_high = radius > freq_threshold
    # Note: We use Amplitude for Mean/Std as that makes more intuitive sense for "strength"
    # reshape for batch processing
    amp_flat = amplitude.reshape(B, -1)
    mask_flat = mask_high.flatten()

    freq_mag_mean = torch.zeros(B, device=device)
    freq_mag_std = torch.zeros(B, device=device)

    if mask_flat.sum() > 0:
        # We can't easily vectorize masking different batch items if counts differ,
        # but here the mask is the same for all images.
        selected = amp_flat[:, mask_flat]
        freq_mag_mean = selected.mean(dim=1)
        freq_mag_std = selected.std(dim=1)

    # 3. 1D Radial Binning (Vectorized Scatter)
    # Round radius to integer bins
    r_bins = radius.round().long().flatten()
    max_bin = int(min(H, W) / 2)  # Nyquist limit

    # We only care up to Nyquist
    valid_mask = r_bins <= max_bin
    r_bins = r_bins[valid_mask]

    # Prepare bins
    power_flat = power.reshape(B, -1)[:, valid_mask]

    # Scatter add to compute sums per bin
    # shape: (B, max_bin + 1)
    sums = torch.zeros((B, max_bin + 1), device=device)
    counts = torch.zeros((B, max_bin + 1), device=device)

    # We expand r_bins to batch dimension for scatter
    r_bins_expanded = r_bins.unsqueeze(0).expand(B, -1)

    sums.scatter_add_(1, r_bins_expanded, power_flat)
    counts.scatter_add_(1, r_bins_expanded, torch.ones_like(power_flat))

    # Average power per radial bin
    # Avoid div by zero
    counts[counts == 0] = 1
    radial_profile = sums / counts

    # 4. Linear Regression on Log-Log
    # We ignore DC (bin 0) and very low freq (often < 2 or 3)
    start_bin = 1
    end_bin = max_bin  # Use full range up to Nyquist

    x = torch.arange(start_bin, end_bin, device=device).float()
    log_x = torch.log(x)

    slopes = []
    for b in range(B):
        y = radial_profile[b, start_bin:end_bin]

        # Filter zeros/NaNs
        mask = (y > 0) & (~torch.isnan(y))

        if mask.sum() < 2:
            slopes.append(float('nan'))
            continue

        log_y = torch.log(y[mask])
        lx = log_x[mask]

        # Slope formula
        n = len(lx)
        num = n * (lx * log_y).sum() - lx.sum() * log_y.sum()
        den = n * (lx ** 2).sum() - lx.sum() ** 2

        slope = num / (den + 1e-8)
        # We return negative slope as positive alpha often, or raw slope.
        # Your code returned raw slope.
        slopes.append(slope.item())

    return torch.tensor(slopes, device=device)

def frequency_mean_api_torch(images, freq_threshold=8, device='cuda'):
    """
    Calculate the following features of grayscale images using PyTorch on GPU:
        - frequency magnitude mean value

    Parameters:
    images (numpy array or torch tensor): Input images. Can be 2D (single image) or 3D (batch of images).
    device (str): Device to use, e.g., 'cuda' or 'cpu'.

    Returns:
        freq_mag_mean: torch.Tensor, (B, ),  magnitude mean for each image
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

# ==========================================
# 2. Main Execution Script
# ==========================================

def calculate_metrics(image_dir, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # 1. Setup
    files = [f for f in os.listdir(image_dir) if f.lower().endswith('.png')]
    if not files:
        print(f"No .png images found in {image_dir}")
        return

    # Dictionary to hold results: {metric_name: {img_name: value}}
    results_map = {
        "luminance_mean": {},
        "contrast": {},
        "entropy": {},
        "sharpness": {},
        "coarseness": {},
        "roughness": {},
        "line_likeness": {},
        "directionality": {},
        "regularity": {},
        "skewness": {},
        "kurtosis": {},
        "spectral_slope": {},
        "freq_mean": {}
    }

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),  # Converts to [0, 1]
    ])

    print(f"Processing {len(files)} images on {device}...")

    # 2. Loop through images
    # We process 1 by 1 to be safe against different image sizes
    for filename in tqdm(files):
        path = os.path.join(image_dir, filename)
        try:
            img = Image.open(path)
            tensor_img = transform(img).unsqueeze(0).to(device)  # (1, 1, H, W)
            tensor_img = tensor_img.squeeze(1)  # (1, H, W) as required by API

            # --- Calculation ---
            results_map["luminance_mean"][filename] = calculate_luminance_mean_pytorch(tensor_img, device).item()
            results_map["contrast"][filename] = contrast_api_pytorch(tensor_img, device).item()
            results_map["entropy"][filename] = calculate_entropy_pytorch(tensor_img, device).item()
            results_map["sharpness"][filename] = simple_sharpness_api_pytorch(tensor_img, device).item()
            results_map["coarseness"][filename] = coarseness_api_pytorch(tensor_img, device).item()
            results_map["roughness"][filename] = roughness_api_pytorch(tensor_img, device).item()
            results_map["line_likeness"][filename] = line_likeness_api_pytorch(tensor_img, device).item()
            results_map["directionality"][filename] = directionality_api_pytorch(tensor_img, device).item()
            results_map["regularity"][filename] = regularity_api_pytorch(tensor_img, device).item()
            results_map["skewness"][filename] = skewness_api_pytorch(tensor_img, device).item()
            results_map["kurtosis"][filename] = kurtosis_api_pytorch(tensor_img, device).item()
            results_map["spectral_slope"][filename] = spectral_slope_api_pytorch(tensor_img, device=device).item()
            results_map["freq_mean"][filename] = frequency_mean_api_torch(tensor_img, device=device).item()

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # 3. Double Confirmation (Normalized STD Check)
    print("\n--- Validation: Normalized STD Check ---")
    warnings_found = False

    for metric_name, data_dict in results_map.items():
        values = np.array(list(data_dict.values()))
        # Filter out NaNs for stats
        values = values[~np.isnan(values)]

        if len(values) == 0:
            print(f"Metric [{metric_name}] returned no valid values.")
            continue

        mean_val = np.mean(values)
        std_val = np.std(values)

        # Normalized STD (Coefficient of Variation)
        # Add epsilon to mean to avoid div by zero
        norm_std = std_val / (abs(mean_val) + 1e-6)

        # Threshold: If variation is extremely low (< 0.1%), the metric might be failing or data is identical
        if norm_std < 0.001:
            print(f"WARNING: Metric [{metric_name}] has very low normalized std ({norm_std:.6f}).")
            print(f"    Mean: {mean_val:.4f}, Std: {std_val:.4f}")
            print(
                f"    Possible causes: Input images are identical, or metric implementation failed (returned constant).")
            warnings_found = True
        else:
            print(f"OK: [{metric_name}] NormSTD={norm_std:.4f}")

    if not warnings_found:
        print("All metrics show reasonable variation.")

    # 4. Save to JSON
    output_path = "wide_image_metrics.json"
    with open(output_path, 'w') as f:
        json.dump(results_map, f, indent=4)

    print(f"\nResults saved to {output_path}")


# Run
if __name__ == "__main__":
    # REPLACE THIS WITH YOUR PATH
    target_path = "../src_data/mcgill_preprocessed"

    if os.path.exists(target_path):
        calculate_metrics(target_path)
    else:
        print(f"Please create the directory '{target_path}' or update the path in the script.")