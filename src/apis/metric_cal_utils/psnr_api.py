import torch
import torch.nn.functional as F
import numpy as np  # For Sobel kernel definition (used in gssim_pytorch)

# For VIF and HaarPSI calculation, the 'piq' library is used.
# Please install it if you haven't: pip install piq
try:
    import piq
except ImportError:
    print("The 'piq' library is not installed. Please install it using 'pip install piq' to use VIF and HaarPSI.")
    piq = None


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


def psnr_pytorch(img1_tensor, img2_tensor, L=1.0, rescale_to_match=True, device=None):
    """
    Calculates Haar Wavelet-Based Perceptual Similarity Index (HaarPSI) using PyTorch and piq.
    Args:
        img1_tensor (torch.Tensor): The reference image tensor. Values expected
                                   in [0,L] or [0,1] if L=1.0.
        img2_tensor (torch.Tensor): The distorted image tensor. Same format as img1_tensor.
        L (float): Dynamic range of the input pixel values (e.g., 255 for uint8, 1.0 for float [0,1]).
        rescale_to_match (bool): If True, resizes img1 to match img2's spatial dimensions.
        device (str, optional): Device to use ('cpu' or 'cuda'). Auto-detects if None.
    Returns:
        torch.Tensor: The HaarPSI score (scalar tensor). Score is typically in [0, 1], higher is better.
    """
    if piq is None:
        raise RuntimeError("piq library is not installed. Please install it: pip install piq")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move to device and ensure float
    img1_tensor = img1_tensor.to(device).float()
    img2_tensor = img2_tensor.to(device).float()

    # Preprocessing: Convert to grayscale
    img1_gray = _convert_to_grayscale_pytorch(img1_tensor, L=L)
    img2_gray = _convert_to_grayscale_pytorch(img2_tensor, L=L)

    # Preprocessing: Resize img1_gray to match img2_gray dimensions if needed
    if rescale_to_match and img1_gray.shape != img2_gray.shape:
        img1_gray_bchw = img1_gray.unsqueeze(0).unsqueeze(0)  # (H,W) -> (1,1,H,W)
        img1_resized_bchw = F.interpolate(img1_gray_bchw,
                                          size=(img2_gray.shape[0], img2_gray.shape[1]),
                                          mode='bicubic',
                                          align_corners=False)
        img1_proc = img1_resized_bchw.squeeze(0).squeeze(0)  # Back to (H,W)
    else:
        img1_proc = img1_gray
    img2_proc = img2_gray

    # Normalize to [0, 1] range for piq.haarpsi
    if L != 1.0:
        img1_proc = img1_proc / L
        img2_proc = img2_proc / L

    # Clamp to [0,1] to ensure values are strictly in this range for piq
    img1_proc = torch.clamp(img1_proc, 0.0, 1.0)
    img2_proc = torch.clamp(img2_proc, 0.0, 1.0)

    # Add batch and channel dimension for piq: (N, C, H, W)
    img1_piq_format = img1_proc.unsqueeze(0).unsqueeze(0)
    img2_piq_format = img2_proc.unsqueeze(0).unsqueeze(0)

    # Calculate psnr using piq.haarpsi
    # piq.psnr expects input tensors to be in the range [0, 1] and data_range=1.0
    psnr_index: torch.Tensor = piq.psnr(img1_piq_format, img2_piq_format, data_range=1.)
    return psnr_index.item()


def psnr_api_torch(recon_img_batch, gt_img_batch, device=None) -> float:
    """
    Calculates HaarPSI score using PyTorch and piq.
    the images in the batch are generated from one same source image by rotating during compound eye simulation

    :param recon_img_batch: Tensor, shape (repeat_times, recon_size, recon_size), images reconstructed from electric signal
    :param gt_img_batch: Tensor, shape (repeat_times, gt_size, gt_size), images ground truth
    :param rescale_to_match: bool, rescale recon_img to gt_img
    :param device: 'cpu' or 'cuda'
    :return:
        float: HaarPSI score
    """
    feature_val = 0.
    for idx in range(recon_img_batch.shape[0]):
        feature_val += psnr_pytorch(recon_img_batch[idx], gt_img_batch[idx], device=device)

    return feature_val / recon_img_batch.shape[0]
