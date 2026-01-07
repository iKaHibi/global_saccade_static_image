import torch
import torch.nn.functional as F
import numpy as np  # For Sobel kernel definition


def _convert_to_grayscale_pytorch(tensor_image, L=1.0):
    """
    Converts an image tensor to grayscale. Expects input as float.
    Handles inputs of shape (H, W), (H, W, 1), (H, W, 3), (1, H, W), (3, H, W).
    Also handles batched inputs (N,C,H,W) or (N,H,W,C) where N=1.
    Output is always (H, W) as float32.
    Args:
        tensor_image (torch.Tensor): Input image tensor.
        L (float): Dynamic range. If L=1.0, values are assumed/clipped to [0,1].
    Returns:
        torch.Tensor: Grayscale image tensor of shape (H, W).
    """
    if not isinstance(tensor_image, torch.Tensor):
        raise TypeError(f"Input must be a torch.Tensor, got {type(tensor_image)}")

    tensor_image = tensor_image.float()  # Ensure float for calculations
    if L == 1.0:  # Clip if values are expected to be in [0,1]
        tensor_image = torch.clamp(tensor_image, 0.0, 1.0)

    # Handle different input dimensions
    if tensor_image.ndim == 2:  # Already (H, W)
        return tensor_image
    elif tensor_image.ndim == 3:
        if tensor_image.shape[0] == 1:  # (1, H, W) -> (H,W)
            return tensor_image.squeeze(0)
        elif tensor_image.shape[0] == 3:  # (C, H, W) e.g. RGB
            weights = torch.tensor([0.2989, 0.5870, 0.1140], device=tensor_image.device, dtype=torch.float32).view(3, 1,
                                                                                                                   1)
            return (tensor_image * weights).sum(dim=0)
        elif tensor_image.shape[2] == 1:  # (H, W, 1) -> (H,W)
            return tensor_image.squeeze(2)
        elif tensor_image.shape[2] == 3:  # (H, W, C) e.g. RGB
            weights = torch.tensor([0.2989, 0.5870, 0.1140], device=tensor_image.device, dtype=torch.float32).view(1, 1,
                                                                                                                   3)
            return (tensor_image * weights).sum(dim=2)
        else:
            raise ValueError(f"Unsupported 3D tensor shape for grayscale conversion: {tensor_image.shape}")
    elif tensor_image.ndim == 4 and tensor_image.shape[0] == 1:  # Batch size 1: (1,C,H,W) or (1,H,W,C)
        if tensor_image.shape[1] == 1:  # (1,1,H,W)
            return tensor_image.squeeze(0).squeeze(0)
        elif tensor_image.shape[1] == 3:  # (1,3,H,W)
            weights = torch.tensor([0.2989, 0.5870, 0.1140], device=tensor_image.device, dtype=torch.float32).view(1, 3,
                                                                                                                   1, 1)
            return (tensor_image * weights).sum(dim=1).squeeze(0)  # Return (H,W)
        elif tensor_image.shape[3] == 1:  # (1,H,W,1)
            return tensor_image.squeeze(0).squeeze(-1)
        elif tensor_image.shape[3] == 3:  # (1,H,W,3)
            # Permute to (1,3,H,W) for easier processing
            tensor_image_chw = tensor_image.permute(0, 3, 1, 2)
            weights = torch.tensor([0.2989, 0.5870, 0.1140], device=tensor_image.device, dtype=torch.float32).view(1, 3,
                                                                                                                   1, 1)
            return (tensor_image_chw * weights).sum(dim=1).squeeze(0)  # Return (H,W)
        else:
            raise ValueError(f"Unsupported 4D tensor shape for grayscale conversion: {tensor_image.shape}")
    else:
        raise ValueError(
            f"Unsupported tensor ndim for grayscale conversion: {tensor_image.ndim}. Expected 2, 3 or 4 (batch_size=1).")


def gssim_pytorch(img1_tensor, img2_tensor, L=1.0, K1=0.01, K2=0.03, rescale_to_match=True, device=None):
    """
    Calculates Gradient-based Structural Similarity (G-SSIM) using PyTorch.
    Args:
        img1_tensor (torch.Tensor): The reference image tensor.
                                   Can be (H,W), (H,W,C), (C,H,W), etc.
        img2_tensor (torch.Tensor): The distorted image tensor.
        L (float): Dynamic range of the original pixel values.
        K1 (float): Constant for C1 calculation.
        K2 (float): Constant for C2 calculation.
        rescale_to_match (bool): If True, resizes img1 to match img2's dimensions.
        device (str, optional): Device to use ('cpu' or 'cuda'). Auto-detects if None.
    Returns:
        torch.Tensor: The G-SSIM score (scalar tensor).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Constants for SSIM
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # Move to device and ensure float
    img1_tensor = img1_tensor.to(device).float()
    img2_tensor = img2_tensor.to(device).float()

    # Preprocessing: Convert to grayscale (output is (H,W) float32 tensor)
    img1_gray = _convert_to_grayscale_pytorch(img1_tensor, L=L)
    img2_gray = _convert_to_grayscale_pytorch(img2_tensor, L=L)

    # Preprocessing: Resize img1_gray to match img2_gray dimensions
    if rescale_to_match and img1_gray.shape != img2_gray.shape:
        # F.interpolate expects (N, C, H, W)
        img1_gray_bchw = img1_gray.unsqueeze(0).unsqueeze(0)
        img1_resized_bchw = F.interpolate(img1_gray_bchw,
                                          size=(img2_gray.shape[0], img2_gray.shape[1]),
                                          mode='bicubic',
                                          align_corners=False)
        img1_proc = img1_resized_bchw.squeeze(0).squeeze(0)
    else:
        img1_proc = img1_gray

    img2_proc = img2_gray  # img2_gray is already the target size

    # Define Sobel kernels for PyTorch
    sobel_kernel_x_np = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_kernel_y_np = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    # Reshape for conv2d: (out_channels, in_channels, kH, kW)
    kernel_x = torch.from_numpy(sobel_kernel_x_np).unsqueeze(0).unsqueeze(0).to(device)
    kernel_y = torch.from_numpy(sobel_kernel_y_np).unsqueeze(0).unsqueeze(0).to(device)

    # Add batch and channel dimension to images: (N, C, H, W)
    img1_bchw = img1_proc.unsqueeze(0).unsqueeze(0)
    img2_bchw = img2_proc.unsqueeze(0).unsqueeze(0)

    # Compute gradients using 2D convolution
    # padding=1 for 3x3 kernel to maintain size ('same' behavior)
    Gx1 = F.conv2d(img1_bchw, kernel_x, padding=1)
    Gy1 = F.conv2d(img1_bchw, kernel_y, padding=1)
    M1 = torch.sqrt(Gx1 ** 2 + Gy1 ** 2).squeeze(0).squeeze(0)  # Back to (H,W)

    Gx2 = F.conv2d(img2_bchw, kernel_x, padding=1)
    Gy2 = F.conv2d(img2_bchw, kernel_y, padding=1)
    M2 = torch.sqrt(Gx2 ** 2 + Gy2 ** 2).squeeze(0).squeeze(0)  # Back to (H,W)

    # Calculate global statistics for gradient magnitude maps M1 and M2
    mu_M1 = torch.mean(M1)
    mu_M2 = torch.mean(M2)

    sigma_M1_sq = torch.var(M1, unbiased=False)  # Population variance (N in denominator)
    sigma_M2_sq = torch.var(M2, unbiased=False)  # Population variance

    # Covariance: E[(M1 - mu_M1)(M2 - mu_M2)]
    sigma_M1M2 = torch.mean((M1 - mu_M1) * (M2 - mu_M2))

    # G-SSIM Calculation
    numerator = (2 * mu_M1 * mu_M2 + C1) * (2 * sigma_M1M2 + C2)
    denominator = (mu_M1 ** 2 + mu_M2 ** 2 + C1) * (sigma_M1_sq + sigma_M2_sq + C2)

    gssim_score = numerator / (denominator + 1e-12)  # Add epsilon for stability

    return gssim_score.item()


def gssim_api_torch(recon_img_batch, gt_img_batch, device=None) -> float:
    """
    Calculates GSSIM score using PyTorch and piq.
    the images in the batch are generated from one same source image by rotating during compound eye simulation

    :param recon_img_batch: Tensor, shape (repeat_times, recon_size, recon_size), images reconstructed from electric signal
    :param gt_img_batch: Tensor, shape (repeat_times, gt_size, gt_size), images ground truth
    :param rescale_to_match: bool, rescale recon_img to gt_img
    :param device: 'cpu' or 'cuda'
    :return:
        float: GSSIM score
    """
    feature_val = 0.
    for idx in range(recon_img_batch.shape[0]):
        feature_val += gssim_pytorch(recon_img_batch[idx], gt_img_batch[idx], device=device)

    return feature_val / recon_img_batch.shape[0]

if __name__ == '__main__':
    print("\nPyTorch G-SSIM Example:")
    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {current_device}")

    # --- Test Case 1: Identical Grayscale tensors (H,W) ---
    ref_gray_t = (torch.rand(128, 128, device=current_device) * 255)
    dist_gray_identical_t = ref_gray_t.clone()
    gssim_id_t = gssim_pytorch(ref_gray_t, dist_gray_identical_t, L=255.0, device=current_device)
    print(f"Identical Grayscale (H,W): G-SSIM = {gssim_id_t.item():.4f}")

    # --- Test Case 2: Grayscale tensors (H,W) with noise ---
    dist_gray_noisy_t = ref_gray_t.clone() + torch.randn(128, 128, device=current_device) * 30
    dist_gray_noisy_t.clamp_(0, 255)
    gssim_noisy_t = gssim_pytorch(ref_gray_t, dist_gray_noisy_t, L=255.0, device=current_device)
    print(f"Noisy Grayscale (H,W): G-SSIM = {gssim_noisy_t.item():.4f}")

    # --- Test Case 3: Color tensors (H,W,C) identical ---
    ref_color_hwc_t = (torch.rand(128, 128, 3, device=current_device) * 255)
    dist_color_hwc_identical_t = ref_color_hwc_t.clone()
    gssim_chwc_id_t = gssim_pytorch(ref_color_hwc_t, dist_color_hwc_identical_t, L=255.0, device=current_device)
    print(f"Identical Color (H,W,C): G-SSIM = {gssim_chwc_id_t.item():.4f}")

    # --- Test Case 4: Color tensors (N,C,H,W) identical, N=1 ---
    ref_color_nchw_t = (torch.rand(1, 3, 128, 128, device=current_device) * 255)
    dist_color_nchw_identical_t = ref_color_nchw_t.clone()
    gssim_cnchw_id_t = gssim_pytorch(ref_color_nchw_t, dist_color_nchw_identical_t, L=255.0, device=current_device)
    print(f"Identical Color (N,C,H,W): G-SSIM = {gssim_cnchw_id_t.item():.4f}")

    # --- Test Case 5: Tensors with different sizes (color H,W,C) ---
    ref_large_t = (torch.rand(150, 160, 3, device=current_device) * 255)
    dist_small_t = (torch.rand(128, 128, 3, device=current_device) * 255)
    gssim_ds_t = gssim_pytorch(ref_large_t, dist_small_t, L=255.0, device=current_device)
    print(f"Different Sizes (ref {ref_large_t.shape}, dist {dist_small_t.shape}): G-SSIM = {gssim_ds_t.item():.4f}")

    # --- Test Case 6: Tensors with [0,1] float range (H,W) ---
    ref_float_t = torch.rand(128, 128, device=current_device, dtype=torch.float32)  # Range [0,1]
    dist_float_noisy_t = torch.clamp(ref_float_t.clone() + torch.randn(128, 128, device=current_device) * 0.15, 0, 1)
    gssim_f_t = gssim_pytorch(ref_float_t, dist_float_noisy_t, L=1.0, device=current_device)
    print(f"Float [0,1] range (H,W): G-SSIM = {gssim_f_t.item():.4f}")

    gssim_f_id_t = gssim_pytorch(ref_float_t, ref_float_t.clone(), L=1.0, device=current_device)
    print(f"Float [0,1] identical (H,W): G-SSIM = {gssim_f_id_t.item():.4f}")
