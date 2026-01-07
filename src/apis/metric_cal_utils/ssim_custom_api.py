import torch
import torch.nn.functional as F
from math import exp


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


def create_window(window_size, channel, device):
    """Generates a 1D Gaussian window."""
    _1D_window = torch.tensor([exp(-(x - window_size // 2) ** 2 / float(2 * 1.5 ** 2)) for x in range(window_size)],
                              device=device)
    _1D_window = _1D_window / _1D_window.sum()
    _1D_window = _1D_window.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim_custom(img1, img2, window, window_size, channel, L=1.0):
    """
    Internal calculation function using reflection padding to match PIQ/Standard behavior.
    """
    padding = window_size // 2

    # Use reflection padding to avoid boundary artifacts (which cause the score drop in zero-padding)
    img1 = F.pad(img1, (padding, padding, padding, padding), mode='reflect')
    img2 = F.pad(img2, (padding, padding, padding, padding), mode='reflect')

    mu1 = F.conv2d(img1, window, groups=channel)
    mu2 = F.conv2d(img2, window, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def _interpolate_by_F(img_2d, target_size):
    img_bchw = img_2d.unsqueeze(0).unsqueeze(0)  # (H,W) -> (1,1,H,W)
    img_resized_bchw = F.interpolate(img_bchw,
                                      size=target_size,
                                      mode='bicubic',
                                      align_corners=False,
                                      # mode='nearest',
                                      )
    img_proc = img_resized_bchw.squeeze(0).squeeze(0)  # Back to (H,W)
    return img_proc

def ssim_pytorch(img1_tensor, img2_tensor, L=1.0, rescale_to_match=True, device=None):
    """
    Calculates SSIM using a custom PyTorch implementation (matching standard behavior).
    Args:
        img1_tensor (torch.Tensor): The reference image tensor. Values expected in [0,L] or [0,1] if L=1.0.
        img2_tensor (torch.Tensor): The distorted image tensor. Same format as img1_tensor.
        L (float): Dynamic range.
        rescale_to_match (bool): If True, resizes img1 to match img2's spatial dimensions.
        device (str, optional): Device to use ('cpu' or 'cuda'). Auto-detects if None.
    Returns:
        float: The SSIM score.
    """
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
        # get the size to resize to
        recon_h, recon_w = img1_gray.shape[:2]
        gt_h, gt_w = img2_gray.shape[:2]

        if (gt_h / recon_h) * (gt_w / recon_w) <= 9:
            img1_proc = _interpolate_by_F(img1_gray, (gt_h, gt_w))
            img2_proc = img2_gray
        else:
            print("!!Using *9 Scaling")
            # in this case, too much interpolation needed to be done in reconstructed image,
            # which add more interpolation algorithm induced information
            resize_h = recon_h * 3
            resize_w = recon_w * 3
            img1_proc = _interpolate_by_F(img1_gray, (resize_h, resize_w))
            img2_proc = _interpolate_by_F(img2_gray, (resize_h, resize_w))
    else:
        img1_proc = img1_gray
        img2_proc = img2_gray

    # Normalize to [0, 1] range for calculation logic if L is not 1.0,
    # but SSIM formula handles L via C1/C2 constants.
    # However, standard convention often computes on [0,1] or [0,255].
    # We pass L explicitly to the calculation function.

    # Add batch and channel dimension: (1, 1, H, W)
    img1_proc = img1_proc.unsqueeze(0).unsqueeze(0)
    img2_proc = img2_proc.unsqueeze(0).unsqueeze(0)

    window_size = 11
    channel = 1
    window = create_window(window_size, channel, device)

    ssim_val = _ssim_custom(img1_proc, img2_proc, window, window_size, channel, L=L)
    return ssim_val.item()


def ssim_api_torch(recon_img_batch, gt_img_batch, device=None) -> float:
    """
    Calculates SSIM score using custom PyTorch implementation.
    the images in the batch are generated from one same source image by rotating during compound eye simulation

    :param recon_img_batch: Tensor, shape (repeat_times, recon_size, recon_size), images reconstructed from electric signal
    :param gt_img_batch: Tensor, shape (repeat_times, gt_size, gt_size), images ground truth
    :param rescale_to_match: bool, rescale recon_img to gt_img
    :param device: 'cpu' or 'cuda'
    :return:
        float: SSIM score
    """
    print("Using Custom SSIM")
    feature_val = 0.
    for idx in range(recon_img_batch.shape[0]):
        feature_val += ssim_pytorch(recon_img_batch[idx], gt_img_batch[idx], device=device)

    return feature_val / recon_img_batch.shape[0]