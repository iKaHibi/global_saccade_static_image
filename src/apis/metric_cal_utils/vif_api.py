import torch
import torch.nn.functional as F
import numpy as np  # For Sobel kernel definition (used in gssim_pytorch)

# For VIF calculation, the 'piq' library is used.
# Please install it if you haven't: pip install piq
import piq


def _convert_to_grayscale_pytorch(tensor_image, L=255.0):
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


def vif_pytorch(img1_tensor, img2_tensor, L=1.0, rescale_to_match=True, device=None):
    """
    Calculates Visual Information Fidelity (VIFp) using PyTorch and the piq library.
    Args:
        img1_tensor (torch.Tensor): The reference image tensor.
                                   Can be (H,W), (H,W,C), (C,H,W), etc. Values expected
                                   in [0,L] or [0,1] if L=1.0.
        img2_tensor (torch.Tensor): The distorted image tensor. Same format as img1_tensor.
        L (float): Dynamic range of the input pixel values (e.g., 255 for uint8, 1.0 for float [0,1]).
        rescale_to_match (bool): If True, resizes img1 to match img2's spatial dimensions.
        device (str, optional): Device to use ('cpu' or 'cuda'). Auto-detects if None.
    Returns:
        torch.Tensor: The VIFp score (scalar tensor). Score is typically in [0, 1], higher is better.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move to device and ensure float
    img1_tensor = img1_tensor.to(device).float()
    img2_tensor = img2_tensor.to(device).float()

    # Preprocessing: Convert to grayscale (output is (H,W) float32 tensor)
    # _convert_to_grayscale_pytorch handles various shapes and uses L for potential clipping if L=1.0.
    # Output of _convert_to_grayscale_pytorch is in the original scale (e.g., 0-255 or 0-1 if L=1.0).
    img1_gray = _convert_to_grayscale_pytorch(img1_tensor, L=L)
    img2_gray = _convert_to_grayscale_pytorch(img2_tensor, L=L)

    # Preprocessing: Resize img1_gray to match img2_gray dimensions if needed
    if rescale_to_match and img1_gray.shape != img2_gray.shape:
        # F.interpolate expects (N, C, H, W)
        img1_gray_bchw = img1_gray.unsqueeze(0).unsqueeze(0)  # (H,W) -> (1,1,H,W)
        img1_resized_bchw = F.interpolate(img1_gray_bchw,
                                          size=(img2_gray.shape[0], img2_gray.shape[1]),
                                          mode='bicubic',
                                          align_corners=False)
        img1_proc = img1_resized_bchw.squeeze(0).squeeze(0)  # Back to (H,W)
    else:
        img1_proc = img1_gray

    img2_proc = img2_gray  # img2_gray is already the target size

    # Normalize to [0, 1] range for piq.vif_p
    # If L is 1.0, _convert_to_grayscale_pytorch already clamped to [0,1] if inputs were outside.
    # If L is > 1.0 (e.g. 255), scale image to [0,1].
    if L != 1.0:
        img1_proc = img1_proc / L
        img2_proc = img2_proc / L

    # Clamp to [0,1] to ensure values are strictly in this range for piq
    img1_proc = torch.clamp(img1_proc, 0.0, 1.0)
    img2_proc = torch.clamp(img2_proc, 0.0, 1.0)

    # Add batch and channel dimension for piq: (N, C, H, W)
    # N=1 (batch size), C=1 (grayscale)
    img1_piq_format = img1_proc.unsqueeze(0).unsqueeze(0)
    img2_piq_format = img2_proc.unsqueeze(0).unsqueeze(0)

    # Calculate VIFp using piq.vif_p
    # piq.vif_p expects input tensors to be in the range [0, 1] and data_range=1.0
    vif_score = piq.vif_p(img1_piq_format, img2_piq_format, data_range=1.0)

    return vif_score.item()


def vif_api_torch(recon_img_batch, gt_img_batch, device=None) -> float:
    """
    Calculates VIF score using PyTorch and piq.
    the images in the batch are generated from one same source image by rotating during compound eye simulation

    :param recon_img_batch: Tensor, shape (repeat_times, recon_size, recon_size), images reconstructed from electric signal
    :param gt_img_batch: Tensor, shape (repeat_times, gt_size, gt_size), images ground truth
    :param rescale_to_match: bool, rescale recon_img to gt_img
    :param device: 'cpu' or 'cuda'
    :return:
        float: VIF score
    """
    feature_val = 0.
    for idx in range(recon_img_batch.shape[0]):
        feature_val += vif_pytorch(recon_img_batch[idx], gt_img_batch[idx], device=device)

    return feature_val / recon_img_batch.shape[0]

if __name__ == '__main__':
    print("\nPyTorch G-SSIM Example:")
    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {current_device}")

    # --- Test Case 1: Identical Grayscale tensors (H,W) ---
    ref_gray_t = (torch.rand(128, 128, device=current_device) * 255)
    dist_gray_identical_t = ref_gray_t.clone()

    # --- Test Case 2: Grayscale tensors (H,W) with noise ---
    dist_gray_noisy_t = ref_gray_t.clone() + torch.randn(128, 128, device=current_device) * 30
    dist_gray_noisy_t.clamp_(0, 255)

    # --- Test Case 3: Color tensors (H,W,C) identical ---
    ref_color_hwc_t = (torch.rand(128, 128, 3, device=current_device) * 255)
    dist_color_hwc_identical_t = ref_color_hwc_t.clone()

    # --- Test Case 4: Color tensors (N,C,H,W) identical, N=1 ---
    ref_color_nchw_t = (torch.rand(1, 3, 128, 128, device=current_device) * 255)
    dist_color_nchw_identical_t = ref_color_nchw_t.clone()

    # --- Test Case 5: Tensors with different sizes (color H,W,C) ---
    ref_large_t = (torch.rand(150, 160, 3, device=current_device) * 255)
    dist_small_t = (torch.rand(128, 128, 3, device=current_device) * 255)

    # --- Test Case 6: Tensors with [0,1] float range (H,W) ---
    ref_float_t = torch.rand(128, 128, device=current_device, dtype=torch.float32)  # Range [0,1]
    dist_float_noisy_t = torch.clamp(ref_float_t.clone() + torch.randn(128, 128, device=current_device) * 0.15, 0, 1)

    print("\nPyTorch VIF Example:")
    # Using the same device as G-SSIM examples
    print(f"Using device for VIF: {current_device}")

    # --- VIF Test Case 1: Identical Grayscale tensors (H,W) ---
    # Reusing ref_gray_t for simplicity, or create new ones if strict separation is needed
    vif_id_t = vif_pytorch(ref_gray_t, dist_gray_identical_t, L=255.0, device=current_device)
    print(f"Identical Grayscale (H,W): VIF = {vif_id_t.item():.4f}")  # Expected ~1.0

    # --- VIF Test Case 2: Grayscale tensors (H,W) with noise ---
    # Reusing dist_gray_noisy_t
    vif_noisy_t = vif_pytorch(ref_gray_t, dist_gray_noisy_t, L=255.0, device=current_device)
    print(f"Noisy Grayscale (H,W): VIF = {vif_noisy_t.item():.4f}")

    # --- VIF Test Case 3: Color tensors (H,W,C) identical ---
    # Reusing ref_color_hwc_t
    vif_chwc_id_t = vif_pytorch(ref_color_hwc_t, dist_color_hwc_identical_t, L=255.0, device=current_device)
    print(f"Identical Color (H,W,C): VIF = {vif_chwc_id_t.item():.4f}")

    # --- VIF Test Case 4: Color tensors (N,C,H,W) identical, N=1 ---
    # Reusing ref_color_nchw_t
    vif_cnchw_id_t = vif_pytorch(ref_color_nchw_t, dist_color_nchw_identical_t, L=255.0, device=current_device)
    print(f"Identical Color (N,C,H,W): VIF = {vif_cnchw_id_t.item():.4f}")

    # --- VIF Test Case 5: Tensors with different sizes (color H,W,C) ---
    # Reusing ref_large_t and dist_small_t
    vif_ds_t = vif_pytorch(ref_large_t, dist_small_t, L=255.0, rescale_to_match=True, device=current_device)
    print(f"Different Sizes (ref {ref_large_t.shape}, dist {dist_small_t.shape}): VIF = {vif_ds_t.item():.4f}")

    # --- VIF Test Case 6: Tensors with [0,1] float range (H,W) ---
    # Reusing ref_float_t and dist_float_noisy_t
    vif_f_t = vif_pytorch(ref_float_t, dist_float_noisy_t, L=1.0, device=current_device)
    print(f"Float [0,1] range (H,W): VIF = {vif_f_t.item():.4f}")

    vif_f_id_t = vif_pytorch(ref_float_t, ref_float_t.clone(), L=1.0, device=current_device)
    print(f"Float [0,1] identical (H,W): VIF = {vif_f_id_t.item():.4f}")  # Expected ~1.0