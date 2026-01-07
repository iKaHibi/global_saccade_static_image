import numpy as np
import torch
import os
import cv2
import random
from pathlib import Path
from typing import Tuple, Dict, Any, List
from skimage.transform import resize
import matplotlib.pyplot as plt

from .e2p_decoding.e2p_lstm import LSTMDecoder
from .photon_sim.photon_filter import PhotonFilter

from .reconstruction_static_img.recon_google_ibp_api_pos_based import google_ibp_api, interpolate_reconstruct_hr_image, pure_ibp_api
from .reconstruction_static_img.recon_ibp_api_grid_based import prepare_data_for_autograd_ibp, solve_ibp_autograd, debug_coordinate_orientation

from ...utils import convert_array2tensor_list, save_image, tensor2uint, todevice

from .random_dir_shift_util import find_first_direction_indices, align_kernels_by_shift

# Get the directory of the current Python file
current_file_dir = os.path.dirname(os.path.abspath(__file__))

def central_crop(image: np.ndarray, ratio: float = 0.9) -> np.ndarray:
    """
    Crops the central portion of an image based on the given ratio.

    Args:
        image: Input image (H, W) or (H, W, C).
        ratio: The fraction of the dimension to keep (default 0.9).

    Returns:
        Cropped image.
    """
    if ratio <= 0 or ratio > 1:
        raise ValueError("Crop ratio must be between 0 and 1.")

    h, w = image.shape[:2]
    new_h, new_w = int(h * ratio), int(w * ratio)

    start_y = (h - new_h) // 2
    start_x = (w - new_w) // 2

    return image[start_y: start_y + new_h, start_x: start_x + new_w]

# Helper functions from the script (refactored to be self-contained or passed)
# It's good practice to place these in utility files, but for simplicity, they are here.
def _sigmoid_electric(x, x0=4000, k=0.001):
    return 1 / (1 + torch.exp(-k * (x - x0)))

from scipy.signal import butter, filtfilt

def _butter_lowpass(cutoff, fs, order=5):
    """
    Design a Butterworth lowpass filter.

    Parameters:
        cutoff (float): Cutoff frequency (in Hz).
        fs (float): Sampling frequency (in Hz).
        order (int): Order of the filter.

    Returns:
        b, a: Numerator and denominator polynomials of the filter.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def _apply_lowpass_filter(tensor, cutoff, fs, order=5):
    """
    Apply a lowpass filter to a PyTorch tensor.

    Parameters:
        tensor (torch.Tensor): Input tensor of shape [batch_size, time_len, 1].
        cutoff (float): Cutoff frequency (in Hz).
        fs (float): Sampling frequency (in Hz).
        order (int): Order of the filter.

    Returns:
        torch.Tensor: Filtered tensor of the same shape as input.
    """
    # Convert tensor to NumPy array
    data = tensor.squeeze(-1).numpy()  # Remove last dimension for filtering

    # Design the filter
    b, a = _butter_lowpass(cutoff, fs, order=order)

    # Apply the filter to each trial in the batch
    filtered_data = np.array([filtfilt(b, a, trial) for trial in data])

    # Convert back to PyTorch tensor and restore the last dimension
    return torch.from_numpy(filtered_data).unsqueeze(-1).to(tensor.device, dtype=tensor.dtype)


def _split_to_batches(res_array: np.ndarray, batch_size: int) -> torch.Tensor:
    T, omm_num = res_array.shape
    assert omm_num % batch_size == 0, f"Ommatidia count ({omm_num}) must be divisible by batch_size ({batch_size})"
    reshaped = res_array.T.reshape(-1, batch_size, T)
    return torch.from_numpy(reshaped).float()


def reconstruction_with_jitter_function(
        rotated_source_sequence: np.ndarray,
        electric_signal: np.ndarray,
        pos_rcd: np.ndarray,
        shift_pix_float_rcd: np.ndarray,
        kernel_data: np.ndarray,
        e2p_model: LSTMDecoder,
        omm_num_sqrt: int,
        shift_pixel_list: List[int],
        device: str,
        config: Dict[str, Any],
        saving_flag: bool = False,
        destination_path: Path = None,
        destination_file_name: str = None,
        with_jitter: bool = True,
        src_img_size: int = 4,
        up_scale: int = 4,
        photon_signal: np.ndarray = None,
        ibp_times: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstructs high-resolution images from simulated electric signals.

    This function wraps the logic of `elec2photon_decoding` and `img_reconstruction`.

    Args:
        rotated_source_sequence (np.ndarray): Ground truth video frames (T, H, W).
        electric_signal (np.ndarray): The simulated electric signal (T_elec, S*S).
        kernel_data (np.ndarray): The corresponding kernels (T, K, K).
        e2p_model (LSTMDecoder): Pre-loaded electric-to-photon decoding model.
        omm_num_sqrt (int): Number of smallest eigenvalues.
        device (str): 'cuda' or 'cpu'.
        config (Dict): Parameters like 'const_batch', 'repeat_times', 'repeat_method'.
        saving_flag (bool): If True, saves the N reconstructed image frames.
        destination_path (Path): Directory to save reconstructed images.
        destination_file_name (str): Base name for saved image files.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
        - Ground truth source images (N, H, W)
        - Reconstructed images (N, H, W)
    """
    # --- 1. Electric to Decoded Photon (from `elec2photon_decoding`) ---
    # photon_frames = np.reshape(photon_signal, (-1, omm_num_sqrt, omm_num_sqrt))
    if photon_signal is None:
        print("Decoding electric signal to photon signal...")
        e2p_model.eval()
        batch_size = config['const_batch']

        # Preprocess electric signal
        electric_batches = _split_to_batches(electric_signal, batch_size)  # (num_omm//B, B, T)

        decoded_photon_list = []
        with torch.no_grad():
            for batch in electric_batches:  # batch: (B, T)
                # Note: Low-pass filtering and normalization would be here.
                electric_input = _apply_lowpass_filter(batch.unsqueeze(2), cutoff=40, fs=1000)
                electric_input = _sigmoid_electric(electric_input)
                electric_input = (electric_input - electric_input.mean()) / electric_input.std()
                electric_input = electric_input.to(device)

                # shape for LSTM: (batch, seq_len, 1)
                decoded_signal = e2p_model(electric_input)
                decoded_photon_list.append(decoded_signal.cpu().squeeze())

        # Reassemble the decoded photon signal
        decoded_photons_batched = torch.cat(decoded_photon_list, dim=0)  # (num_omm // B, B, T)
        decoded_photon_signal = decoded_photons_batched.view(electric_signal.shape[1], -1).T.numpy()  # (T_elec, S*S)
        decoded_photon_signal = decoded_photon_signal.reshape(decoded_photon_signal.shape[0],
                                                            int(np.sqrt(decoded_photon_signal.shape[1])),
                                                            int(np.sqrt(decoded_photon_signal.shape[1])))
        # decoded_photon_signal = decoded_photon_signal[::10, decoded_photon_signal.shape[1], decoded_photon_signal.shape[2]]  # (T_photon, S, S)
    else:
        print("Using photon signal")
        decoded_photon_signal = np.reshape(photon_signal, (-1, omm_num_sqrt, omm_num_sqrt)) # (T_photon, S, S)


    # --- 2. Decoded Photon to Reconstructed Image (from `img_reconstruction`) ---
    print("Reconstructing images from decoded photons...")
    sample_num = 1

    if with_jitter:
        sampled_indices_dict = find_first_direction_indices(shift_pix_float_rcd, shift_pixel_list, start_idx=50, debug_flag=False)
        sampled_indices = []
        for _, v in sampled_indices_dict.items():
            if v is not None:
                sampled_indices.append(v)
        sampled_indices = np.array(sampled_indices)
    else:
        sampled_indices = np.array([63])

    gt_images = rotated_source_sequence[sampled_indices]
    chosen_pos_rcd = pos_rcd[sampled_indices]

    if photon_signal is None:
        # Use decoded photons at corresponding time steps (1000fps -> 100fps)
        lr_images = decoded_photon_signal[[idx * 10 for idx in sampled_indices]]
    else:
        lr_images = decoded_photon_signal[[idx for idx in sampled_indices]]
    kernel_arr = kernel_data[sampled_indices]

    origin_hr_size = gt_images[0].shape[0]
    lr_size = lr_images[0].shape[0]
    shifted_pix_float_origin = shift_pix_float_rcd[sampled_indices]


    if config["reconstruction_mode"] == "only_interp":
        print("using only interpolation mode...")
        initial_guess_torch = interpolate_reconstruct_hr_image(lr_images, chosen_pos_rcd, src_img_size,
                                                               up_scale=up_scale, method="linear")
        recon_img_torch = initial_guess_torch.transpose(dim0=1, dim1=0).float()
    else:
        initial_guess_torch = interpolate_reconstruct_hr_image(lr_images, chosen_pos_rcd, src_img_size,
                                                               up_scale=up_scale,)
        chosen_pos_rcd = np.resize(chosen_pos_rcd, (chosen_pos_rcd.shape[0], lr_size, lr_size, chosen_pos_rcd.shape[-1]))

        # debug_coordinate_orientation(chosen_pos_rcd)

        lr_t, grid_t, k_recon = prepare_data_for_autograd_ibp(lr_images, chosen_pos_rcd, kernel_arr, origin_hr_size, origin_hr_size, lr_size*4, lr_size*4)
        initial_guess = initial_guess_torch.transpose(dim0=1, dim1=0).float()
        if initial_guess.dim() == 2:
            initial_guess = initial_guess.unsqueeze(0).unsqueeze(0)
        elif initial_guess.dim() == 3:
            initial_guess = initial_guess.unsqueeze(0)

        recon_img_torch = solve_ibp_autograd(lr_t, grid_t, k_recon, initial_guess)

    recon_img_np = central_crop(tensor2uint(recon_img_torch))

    if saving_flag:
        for i in range(sample_num):
            # saving photon signal
            decoded_photons_save_path = destination_path / 'sampled_photons' / destination_file_name
            decoded_photons_save_path.mkdir(parents=True, exist_ok=True)
            save_image(np.array(lr_images[i], dtype=np.uint8), decoded_photons_save_path / f"{i:04d}.png", resize_size=160)

            # saving gt img
            gt_img_save_path = destination_path / 'gt_img' / destination_file_name
            gt_img_save_path.mkdir(parents=True, exist_ok=True)
            save_image(np.array(gt_images[i], dtype=np.uint8), gt_img_save_path / f"{i:04d}.png") # TODO: check gt_image range

            # saving reconstructed image
            save_path = destination_path / 'recon_img' / destination_file_name
            save_path.mkdir(parents=True, exist_ok=True)
            save_image(np.array(recon_img_np, dtype=np.uint8), save_path / f"{i:04d}.png")

    return gt_images[0], recon_img_np