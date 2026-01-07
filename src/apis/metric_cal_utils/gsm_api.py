import torch
import torch.nn.functional as F
import numpy as np  # For initial kernel definition


def _convert_to_grayscale_pytorch(tensor_image, L=255.0):
    """
    Converts an image tensor to grayscale. Expects input as float.
    Handles inputs of shape (H, W), (H, W, 1), (H, W, 3), (1, H, W), (3, H, W).
    Output is always (H, W) as float32.
    Args:
        tensor_image (torch.Tensor): Input image tensor.
        L (float): Dynamic range. If L=1.0, values are assumed/clipped to [0,1].
    Returns:
        torch.Tensor: Grayscale image tensor of shape (H, W).
    """
    if not isinstance(tensor_image, torch.Tensor):
        raise TypeError(f"Input must be a torch.Tensor, got {type(tensor_image)}")

    # Ensure float for calculations.
    # If original was uint8 and L=255, it's fine. If float and L=1, also fine.
    tensor_image = tensor_image.float()
    if L == 1.0:  # Clip if values are expected to be in [0,1]
        tensor_image = torch.clamp(tensor_image, 0.0, 1.0)

    if tensor_image.ndim == 2:  # Already (H, W)
        return tensor_image
    elif tensor_image.ndim == 3:
        if tensor_image.shape[0] == 1:  # (1, H, W) -> (H,W)
            return tensor_image.squeeze(0)
        elif tensor_image.shape[0] == 3:  # (C, H, W) e.g. RGB
            # Y = 0.2989 * R + 0.5870 * G + 0.1140 * B
            weights = torch.tensor([0.2989, 0.5870, 0.1140], device=tensor_image.device, dtype=torch.float32).view(3, 1,
                                                                                                                   1)
            grayscale_image = (tensor_image * weights).sum(dim=0)
            return grayscale_image
        elif tensor_image.shape[2] == 1:  # (H, W, 1) -> (H,W)
            return tensor_image.squeeze(2)
        elif tensor_image.shape[2] == 3:  # (H, W, C) e.g. RGB
            weights = torch.tensor([0.2989, 0.5870, 0.1140], device=tensor_image.device, dtype=torch.float32).view(1, 1,
                                                                                                                   3)
            grayscale_image = (tensor_image * weights).sum(dim=2)
            return grayscale_image
        else:
            raise ValueError(f"Unsupported 3D tensor shape for grayscale conversion: {tensor_image.shape}")
    elif tensor_image.ndim == 4 and tensor_image.shape[0] == 1:  # (N,C,H,W) or (N,H,W,C)
        if tensor_image.shape[1] == 1:  # (N,1,H,W)
            return tensor_image.squeeze(0).squeeze(0)
        elif tensor_image.shape[1] == 3:  # (N,3,H,W)
            weights = torch.tensor([0.2989, 0.5870, 0.1140], device=tensor_image.device, dtype=torch.float32).view(1, 3,
                                                                                                                   1, 1)
            grayscale_image = (tensor_image * weights).sum(dim=1, keepdim=True)
            return grayscale_image.squeeze(0).squeeze(0)  # Return (H,W)
        elif tensor_image.shape[3] == 1:  # (N,H,W,1)
            return tensor_image.squeeze(0).squeeze(-1)
        elif tensor_image.shape[3] == 3:  # (N,H,W,3)
            weights = torch.tensor([0.2989, 0.5870, 0.1140], device=tensor_image.device, dtype=torch.float32).view(1, 1,
                                                                                                                   1, 3)
            grayscale_image = (tensor_image * weights).sum(dim=3, keepdim=True)
            return grayscale_image.squeeze(0).squeeze(-1)  # Return (H,W)
        else:
            raise ValueError(f"Unsupported 4D tensor shape for grayscale conversion: {tensor_image.shape}")
    else:
        raise ValueError(
            f"Unsupported tensor ndim for grayscale conversion: {tensor_image.ndim}. Expected 2, 3 or 4 (batch_size=1).")


def gmsd_pytorch(img1_tensor, img2_tensor, L=1.0, c_const=None, rescale_to_match=True, downsample=False, device=None):
    """
    Calculates Gradient Magnitude Similarity Deviation (GMSD) and Mean GMS using PyTorch.
    Args:
        img1_tensor (torch.Tensor): The reference image tensor.
                                   Can be (H,W), (H,W,C), (C,H,W), etc.
        img2_tensor (torch.Tensor): The distorted image tensor.
        L (float): Dynamic range of pixel values.
        c_const (float, optional): Stability constant for GMS calculation.
                                   If None, defaults to 170.0 for L=255.0, or 0.0026 for L=1.0.
        rescale_to_match (bool): If True, resizes img1 to match img2's dimensions.
        downsample (bool): If True, downsamples images by a factor of 2.
        device (str, optional): Device to use ('cpu' or 'cuda'). Auto-detects if None.
    Returns:
        tuple: (mean_gms, gmsd_score) as torch scalar tensors.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if c_const is None:
        if L == 255.0:
            c_const = 170.0
        elif L == 1.0:
            c_const = 0.0026
        else:
            raise ValueError("L must be 255.0 or 1.0 if c_const is not provided, or provide c_const.")

    # Move to device and ensure float
    img1_tensor = img1_tensor.to(device).float()
    img2_tensor = img2_tensor.to(device).float()

    # Preprocessing: Convert to grayscale
    img1_gray = _convert_to_grayscale_pytorch(img1_tensor, L=L)  # Now (H,W)
    img2_gray = _convert_to_grayscale_pytorch(img2_tensor, L=L)  # Now (H,W)

    # Preprocessing: Resize img1_gray to match img2_gray dimensions
    if rescale_to_match and img1_gray.shape != img2_gray.shape:
        # print(f"Warning: Image dimensions differ. Resizing reference image img1 ({img1_gray.shape}) to match distorted image img2 ({img2_gray.shape}).")
        # F.interpolate expects (N, C, H, W)
        img1_gray_bchw = img1_gray.unsqueeze(0).unsqueeze(0)
        img1_resized_bchw = F.interpolate(img1_gray_bchw,
                                          size=(img2_gray.shape[0], img2_gray.shape[1]),
                                          mode='bicubic',
                                          align_corners=False)
        img1_gray = img1_resized_bchw.squeeze(0).squeeze(0)

    # Preprocessing: Optional downsampling
    if downsample:
        new_h_1, new_w_1 = img1_gray.shape[0] // 2, img1_gray.shape[1] // 2
        new_h_2, new_w_2 = img2_gray.shape[0] // 2, img2_gray.shape[1] // 2

        if new_h_1 == 0 or new_w_1 == 0 or new_h_2 == 0 or new_w_2 == 0:
            print(
                f"Warning: Image dimensions ({img1_gray.shape}, {img2_gray.shape}) too small for downsampling by 2. Skipping downsampling.")
            img1_proc = img1_gray
            img2_proc = img2_gray
        else:
            # F.interpolate expects (N,C,H,W)
            img1_proc = F.interpolate(img1_gray.unsqueeze(0).unsqueeze(0), size=(new_h_1, new_w_1), mode='bicubic',
                                      align_corners=False).squeeze(0).squeeze(0)
            img2_proc = F.interpolate(img2_gray.unsqueeze(0).unsqueeze(0), size=(new_h_2, new_w_2), mode='bicubic',
                                      align_corners=False).squeeze(0).squeeze(0)
    else:
        img1_proc = img1_gray
        img2_proc = img2_gray

    # Define Prewitt kernels for PyTorch
    kernel_x_np = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    kernel_y_np = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)

    # Reshape for conv2d: (out_channels, in_channels, kH, kW)
    kernel_x = torch.from_numpy(kernel_x_np).unsqueeze(0).unsqueeze(0).to(device)
    kernel_y = torch.from_numpy(kernel_y_np).unsqueeze(0).unsqueeze(0).to(device)

    # Add batch and channel dimension to images: (N, C, H, W)
    img1_bchw = img1_proc.unsqueeze(0).unsqueeze(0)
    img2_bchw = img2_proc.unsqueeze(0).unsqueeze(0)

    # Compute gradients
    # padding='same' equivalent for 3x3 kernel is padding=1
    grad_x_r = F.conv2d(img1_bchw, kernel_x, padding=1)
    grad_y_r = F.conv2d(img1_bchw, kernel_y, padding=1)
    m_r = torch.sqrt(grad_x_r ** 2 + grad_y_r ** 2).squeeze(0).squeeze(0)  # Back to (H,W)

    grad_x_d = F.conv2d(img2_bchw, kernel_x, padding=1)
    grad_y_d = F.conv2d(img2_bchw, kernel_y, padding=1)
    m_d = torch.sqrt(grad_x_d ** 2 + grad_y_d ** 2).squeeze(0).squeeze(0)  # Back to (H,W)

    # Calculate GMS map
    gms_map_numerator = 2 * m_r * m_d + c_const
    gms_map_denominator = m_r ** 2 + m_d ** 2 + c_const
    gms_map = gms_map_numerator / (gms_map_denominator + 1e-12)  # Adding small epsilon

    # Mean GMS
    mean_gms = torch.mean(gms_map)

    # GMSD
    # torch.std by default uses N-1 denominator (unbiased). For N, set correction=0.
    # Paper implies population std dev (denominator N).

    return mean_gms.item()


# def gmsd_pytorch(img1_tensor, img2_tensor, L=1.0, c_const=None, rescale_to_match=True, downsample=False, device=None)
def gms_api_torch(recon_img_batch, gt_img_batch, device=None) -> float:
    """
    Calculates GMS score using PyTorch and piq.
    the images in the batch are generated from one same source image by rotating during compound eye simulation

    :param recon_img_batch: Tensor, shape (repeat_times, recon_size, recon_size), images reconstructed from electric signal
    :param gt_img_batch: Tensor, shape (repeat_times, gt_size, gt_size), images ground truth
    :param rescale_to_match: bool, rescale recon_img to gt_img
    :param device: 'cpu' or 'cuda'
    :return:
        float: GMS score
    """
    feature_val = 0.
    for idx in range(recon_img_batch.shape[0]):
        feature_val += gmsd_pytorch(recon_img_batch[idx], gt_img_batch[idx], device=device)

    return feature_val / recon_img_batch.shape[0]

if __name__ == '__main__':
    print("\nPyTorch GMSD Example:")
    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {current_device}")

    # --- Test Case 1: Identical Grayscale tensors (H,W) ---
    ref_gray_t = (torch.rand(128, 128, device=current_device) * 255)
    dist_gray_identical_t = ref_gray_t.clone()
    mean_gms_id_t, gmsd_id_t = gmsd_pytorch(ref_gray_t, dist_gray_identical_t, L=255.0, device=current_device)
    print(f"Identical Grayscale (H,W): Mean GMS = {mean_gms_id_t.item():.4f}, GMSD = {gmsd_id_t.item():.4f}")

    # --- Test Case 2: Grayscale tensors (H,W) with noise ---
    dist_gray_noisy_t = ref_gray_t.clone() + torch.randn(128, 128, device=current_device) * 20
    dist_gray_noisy_t.clamp_(0, 255)
    mean_gms_noisy_t, gmsd_noisy_t = gmsd_pytorch(ref_gray_t, dist_gray_noisy_t, L=255.0, device=current_device)
    print(f"Noisy Grayscale (H,W): Mean GMS = {mean_gms_noisy_t.item():.4f}, GMSD = {gmsd_noisy_t.item():.4f}")

    # --- Test Case 3: Color tensors (H,W,C) identical ---
    ref_color_hwc_t = (torch.rand(128, 128, 3, device=current_device) * 255)
    dist_color_hwc_identical_t = ref_color_hwc_t.clone()
    mean_gms_chwc_id_t, gmsd_chwc_id_t = gmsd_pytorch(ref_color_hwc_t, dist_color_hwc_identical_t, L=255.0,
                                                      device=current_device)
    print(f"Identical Color (H,W,C): Mean GMS = {mean_gms_chwc_id_t.item():.4f}, GMSD = {gmsd_chwc_id_t.item():.4f}")

    # --- Test Case 4: Color tensors (N,C,H,W) identical, N=1 ---
    ref_color_nchw_t = (torch.rand(1, 3, 128, 128, device=current_device) * 255)
    dist_color_nchw_identical_t = ref_color_nchw_t.clone()
    mean_gms_cnchw_id_t, gmsd_cnchw_id_t = gmsd_pytorch(ref_color_nchw_t, dist_color_nchw_identical_t, L=255.0,
                                                        device=current_device)
    print(
        f"Identical Color (N,C,H,W): Mean GMS = {mean_gms_cnchw_id_t.item():.4f}, GMSD = {gmsd_cnchw_id_t.item():.4f}")

    # --- Test Case 5: Tensors with different sizes (color H,W,C) ---
    ref_large_t = (torch.rand(150, 160, 3, device=current_device) * 255)
    dist_small_t = (torch.rand(128, 128, 3, device=current_device) * 255)
    mean_gms_ds_t, gmsd_ds_t = gmsd_pytorch(ref_large_t, dist_small_t, L=255.0, device=current_device)
    print(
        f"Different Sizes (ref {ref_large_t.shape}, dist {dist_small_t.shape}): Mean GMS = {mean_gms_ds_t.item():.4f}, GMSD = {gmsd_ds_t.item():.4f}")

    # --- Test Case 6: Tensors with [0,1] float range (H,W) ---
    ref_float_t = torch.rand(128, 128, device=current_device)  # Range [0,1]
    dist_float_noisy_t = torch.clamp(ref_float_t.clone() + torch.randn(128, 128, device=current_device) * 0.1, 0, 1)
    mean_gms_f_t, gmsd_f_t = gmsd_pytorch(ref_float_t, dist_float_noisy_t, L=1.0, device=current_device)
    print(f"Float [0,1] range (H,W): Mean GMS = {mean_gms_f_t.item():.4f}, GMSD = {gmsd_f_t.item():.4f}")

    mean_gms_f_id_t, gmsd_f_id_t = gmsd_pytorch(ref_float_t, ref_float_t.clone(), L=1.0, device=current_device)
    print(f"Float [0,1] identical (H,W): Mean GMS = {mean_gms_f_id_t.item():.4f}, GMSD = {gmsd_f_id_t.item():.4f}")
