import torch
import torch.nn.functional as F
import numpy as np  # For Sobel kernel definition (used in gssim_pytorch)

# For VIF, HaarPSI, and LPIPS calculation, the 'piq' library is used.
# Please install it if you haven't: pip install piq
try:
    import piq
except ImportError:
    print(
        "The 'piq' library is not installed. Please install it using 'pip install piq' to use VIF, HaarPSI, and LPIPS.")
    piq = None


def _convert_to_grayscale_pytorch(tensor_image, L=255.0):
    """
    Converts an image tensor to grayscale. Expects input as float.
    Handles inputs of shape (H, W), (H, W, 1), (H, W, 3), (1, H, W), (3, H, W).
    Also handles batched inputs (N,C,H,W) or (N,H,W,C) where N=1 (for batched N>1, it processes the first image in batch).
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
    elif tensor_image.ndim == 4:  # Batch size N
        # If batched, process the first image for this helper, or adapt if batch processing is needed here
        img_to_convert = tensor_image[0] if tensor_image.shape[0] > 1 else tensor_image.squeeze(0)

        if img_to_convert.shape[0] == 1:  # (1,H,W) after potential squeeze
            return img_to_convert.squeeze(0)
        elif img_to_convert.shape[0] == 3:  # (3,H,W) after potential squeeze
            weights = torch.tensor([0.2989, 0.5870, 0.1140], device=img_to_convert.device, dtype=torch.float32).view(3,
                                                                                                                     1,
                                                                                                                     1)
            return (img_to_convert * weights).sum(dim=0)
        # Handling (H,W,C) case if squeeze(0) resulted in it (e.g. from (1,H,W,C))
        elif img_to_convert.ndim == 3 and img_to_convert.shape[2] == 1:  # (H,W,1)
            return img_to_convert.squeeze(2)
        elif img_to_convert.ndim == 3 and img_to_convert.shape[2] == 3:  # (H,W,3)
            weights = torch.tensor([0.2989, 0.5870, 0.1140], device=img_to_convert.device, dtype=torch.float32).view(1,
                                                                                                                     1,
                                                                                                                     3)
            return (img_to_convert * weights).sum(dim=2)
        else:
            raise ValueError(f"Unsupported 4D tensor shape for grayscale conversion: {tensor_image.shape}")
    else:
        raise ValueError(
            f"Unsupported tensor ndim for grayscale conversion: {tensor_image.ndim}. Expected 2, 3 or 4.")


def _prepare_image_for_lpips(tensor_image, device):
    """
    Prepares an image tensor for LPIPS calculation.
    Ensures 3 channels (RGB), (N,C,H,W) format, and moves to device.
    Input tensor_image can have various shapes.
    Normalization to [0,1] is NOT done here.
    Args:
        tensor_image (torch.Tensor): Input image tensor.
        device (torch.device): Target device.
    Returns:
        torch.Tensor: Processed image tensor of shape (N, 3, H, W).
    """
    if not isinstance(tensor_image, torch.Tensor):
        raise TypeError(f"Input must be a torch.Tensor, got {type(tensor_image)}")

    img = tensor_image.to(device).float()

    # Handle different input dimensions to get (N, C, H, W)
    if img.ndim == 2:  # (H, W) -> Grayscale
        img = img.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif img.ndim == 3:
        if img.shape[0] == 1:  # (1, H, W) -> Grayscale (channel first)
            img = img.unsqueeze(0)  # (1, 1, H, W)
        elif img.shape[0] == 3:  # (3, H, W) -> Color (channel first)
            img = img.unsqueeze(0)  # (1, 3, H, W)
        elif img.shape[2] == 1:  # (H, W, 1) -> Grayscale (channel last)
            img = img.permute(2, 0, 1).unsqueeze(0)  # (1, 1, H, W)
        elif img.shape[2] == 3:  # (H, W, 3) -> Color (channel last)
            img = img.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        else:
            raise ValueError(f"Unsupported 3D tensor shape for LPIPS preparation: {img.shape}")
    elif img.ndim == 4:  # Potentially (N, C, H, W) or (N, H, W, C)
        if img.shape[3] == 1:  # (N, H, W, 1) -> Grayscale (channels last)
            img = img.permute(0, 3, 1, 2)  # (N, 1, H, W)
        elif img.shape[3] == 3:  # (N, H, W, 3) -> Color (channels last)
            img = img.permute(0, 3, 1, 2)  # (N, 3, H, W)
        elif img.shape[1] == 1 or img.shape[1] == 3:  # Already (N, C, H, W) (channels first)
            pass  # No change needed
        else:
            raise ValueError(f"Unsupported 4D tensor shape for LPIPS: {img.shape}. Expects C=1 or C=3 at dim 1 or 3.")
    else:
        raise ValueError(f"Unsupported tensor ndim for LPIPS: {img.ndim}. Expected 2, 3, or 4.")

    # Ensure 3 channels (C=3) for LPIPS
    if img.shape[1] == 1:  # If single channel (grayscale)
        img = img.repeat(1, 3, 1, 1)  # Repeat channel to make it (N, 3, H, W)
    elif img.shape[1] != 3:
        raise ValueError(f"Image must have 1 or 3 channels after initial processing for LPIPS, got {img.shape[1]}")

    return img  # Shape (N, 3, H, W)


def lpips_pytorch(img1_tensor, img2_tensor, L=1.0, rescale_to_match=True,
                  net_type='alex', version='0.1', device=None):
    """
    Calculates Learned Perceptual Image Patch Similarity (LPIPS) using PyTorch and piq.
    Args:
        img1_tensor (torch.Tensor): The reference image tensor.
        img2_tensor (torch.Tensor): The distorted image tensor.
        L (float): Dynamic range of the input pixel values (e.g., 255 for uint8, 1.0 for float [0,1]).
                   Used to normalize images to [0,1] range for LPIPS.
        rescale_to_match (bool): If True, resizes img1 to match img2's spatial dimensions.
        net_type (str): Type of network to use ('alex', 'vgg', 'squeeze'). Default is 'alex'.
        version (str): Version of the LPIPS model. Default is '0.1'.
        device (torch.device, optional): Device to use. Auto-detects if None.
    Returns:
        torch.Tensor: The LPIPS score (scalar tensor). Lower is better.
    """
    if piq is None:
        raise RuntimeError("piq library is not installed. Please install it: pip install piq")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare images: ensure (N,3,H,W), on device, float
    img1_p = _prepare_image_for_lpips(img1_tensor, device)
    img2_p = _prepare_image_for_lpips(img2_tensor, device)

    # Rescale if necessary (must be done on (N,C,H,W) format)
    if rescale_to_match and img1_p.shape[2:] != img2_p.shape[2:]:
        # Ensure N=1 for interpolate if not already, or that interpolate handles N>1
        # For LPIPS, typically comparing two images, so N=1 is common for img1_p, img2_p
        img1_p = F.interpolate(img1_p, size=img2_p.shape[2:], mode='bicubic', align_corners=False)

    # Normalize to [0, 1] range as expected by piq.LPIPS
    if L != 1.0:
        img1_p = img1_p / L
        img2_p = img2_p / L

    img1_p = torch.clamp(img1_p, 0.0, 1.0)
    img2_p = torch.clamp(img2_p, 0.0, 1.0)

    # Instantiate LPIPS model
    # replace_pooling=True is often recommended for better correlation.
    # reduction='mean' is default and gives a single score per image pair.
    lpips_metric = piq.LPIPS(replace_pooling=True).to(device)
    lpips_metric.eval()  # Set to evaluation mode

    # Calculate LPIPS score
    # piq.LPIPS expects inputs in [0,1] range.
    # It handles batch internally if N > 1.
    score = lpips_metric(img1_p, img2_p)
    return score.item()


def lpips_api_torch(recon_img_batch, gt_img_batch, device=None) -> float:
    """
    Calculates LPIPS score using PyTorch and piq.
    the images in the batch are generated from one same source image by rotating during compound eye simulation

    :param recon_img_batch: Tensor, shape (repeat_times, recon_size, recon_size), images reconstructed from electric signal
    :param gt_img_batch: Tensor, shape (repeat_times, gt_size, gt_size), images ground truth
    :param rescale_to_match: bool, rescale recon_img to gt_img
    :param device: 'cpu' or 'cuda'
    :return:
        float: LPIPS score
    """
    feature_val = 0.
    for idx in range(recon_img_batch.shape[0]):
        feature_val += lpips_pytorch(recon_img_batch[idx], gt_img_batch[idx], device=device)

    return feature_val / recon_img_batch.shape[0]

if __name__ == '__main__':
    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {current_device}")

    # --- Common Test Data ---
    # Grayscale, L=255
    ref_gray_t_255 = (torch.rand(128, 128, device=current_device) * 255)
    dist_gray_identical_t_255 = ref_gray_t_255.clone()
    dist_gray_noisy_t_255 = torch.clamp(ref_gray_t_255.clone() + torch.randn(128, 128, device=current_device) * 30, 0,
                                        255)

    # Color HWC, L=255
    ref_color_hwc_t_255 = (torch.rand(128, 128, 3, device=current_device) * 255)
    dist_color_hwc_identical_t_255 = ref_color_hwc_t_255.clone()
    dist_color_hwc_noisy_t_255 = torch.clamp(
        ref_color_hwc_t_255.clone() + torch.randn(128, 128, 3, device=current_device) * 30, 0, 255)

    # Color NCHW, L=255
    ref_color_nchw_t_255 = (torch.rand(1, 3, 128, 128, device=current_device) * 255)
    dist_color_nchw_identical_t_255 = ref_color_nchw_t_255.clone()
    dist_color_nchw_noisy_t_255 = torch.clamp(
        ref_color_nchw_t_255.clone() + torch.randn(1, 3, 128, 128, device=current_device) * 30, 0, 255)

    # Different sizes, L=255 (Color HWC)
    ref_large_color_hwc_t_255 = (torch.rand(150, 160, 3, device=current_device) * 255)
    dist_small_color_hwc_t_255 = (torch.rand(128, 128, 3, device=current_device) * 255)

    # Float [0,1] range, L=1.0 (Color HWC)
    ref_color_float_hwc_t_1 = torch.rand(128, 128, 3, device=current_device, dtype=torch.float32)
    dist_color_float_hwc_identical_t_1 = ref_color_float_hwc_t_1.clone()
    dist_color_float_hwc_noisy_t_1 = torch.clamp(
        ref_color_float_hwc_t_1.clone() + torch.randn(128, 128, 3, device=current_device) * 0.15, 0, 1)

    # Float [0,1] range, L=1.0 (Grayscale for LPIPS test)
    ref_gray_float_t_1 = torch.rand(128, 128, device=current_device, dtype=torch.float32)
    dist_gray_float_identical_t_1 = ref_gray_float_t_1.clone()

    if piq is not None:
        print("\n--- PyTorch LPIPS Examples ---")
        # Test 1: Identical Color images (HWC, L=255)
        lpips_chwc_id_255 = lpips_pytorch(ref_color_hwc_t_255, dist_color_hwc_identical_t_255, L=255.0)
        print(f"Identical Color (H,W,C, L=255): LPIPS = {lpips_chwc_id_255.item():.4f}")  # Expected near 0

        # Test 2: Noisy Color images (HWC, L=255)
        lpips_chwc_noisy_255 = lpips_pytorch(ref_color_hwc_t_255, dist_color_hwc_noisy_t_255, L=255.0)
        print(f"Noisy Color (H,W,C, L=255): LPIPS = {lpips_chwc_noisy_255.item():.4f}")

        # Test 3: Identical Color images (NCHW, L=255)
        lpips_nchw_id_255 = lpips_pytorch(ref_color_nchw_t_255, dist_color_nchw_identical_t_255, L=255.0)
        print(f"Identical Color (N,C,H,W, L=255): LPIPS = {lpips_nchw_id_255.item():.4f}")

        # Test 4: Noisy Color images (NCHW, L=255)
        lpips_nchw_noisy_255 = lpips_pytorch(ref_color_nchw_t_255, dist_color_nchw_noisy_t_255, L=255.0)
        print(f"Noisy Color (N,C,H,W, L=255): LPIPS = {lpips_nchw_noisy_255.item():.4f}")

        # Test 5: Different sizes (Color HWC, L=255), rescale=True
        lpips_ds_255 = lpips_pytorch(ref_large_color_hwc_t_255, dist_small_color_hwc_t_255, L=255.0,
                                     rescale_to_match=True)
        print(f"Different Sizes (Color HWC, L=255, Rescaled): LPIPS = {lpips_ds_255.item():.4f}")

        # Test 6: Identical Color images (HWC, L=1.0, float input)
        lpips_chwc_id_1 = lpips_pytorch(ref_color_float_hwc_t_1, dist_color_float_hwc_identical_t_1, L=1.0)
        print(f"Identical Color (H,W,C, L=1.0): LPIPS = {lpips_chwc_id_1.item():.4f}")

        # Test 7: Noisy Color images (HWC, L=1.0, float input)
        lpips_chwc_noisy_1 = lpips_pytorch(ref_color_float_hwc_t_1, dist_color_float_hwc_noisy_t_1, L=1.0)
        print(f"Noisy Color (H,W,C, L=1.0): LPIPS = {lpips_chwc_noisy_1.item():.4f}")

        # Test 8: Identical Grayscale images (H,W, L=1.0, float input) -> converted to 3-channel by LPIPS func
        lpips_gray_id_1 = lpips_pytorch(ref_gray_float_t_1, dist_gray_float_identical_t_1, L=1.0)
        print(f"Identical Grayscale (H,W, L=1.0, auto-3ch): LPIPS = {lpips_gray_id_1.item():.4f}")

        # Test 9: Using VGG network for LPIPS
        try:
            lpips_vgg_id_255 = lpips_pytorch(ref_color_hwc_t_255, dist_color_hwc_identical_t_255, L=255.0,
                                             net_type='vgg')
            print(f"Identical Color (H,W,C, L=255, VGG): LPIPS = {lpips_vgg_id_255.item():.4f}")
        except Exception as e:
            print(f"Could not run LPIPS with VGG (อาจจะต้องดาวน์โหลด weights หรือ model ไม่พร้อมใช้งาน): {e}")

    else:
        print("\nSkipping VIF, HaarPSI, and LPIPS examples because 'piq' library is not installed.")

