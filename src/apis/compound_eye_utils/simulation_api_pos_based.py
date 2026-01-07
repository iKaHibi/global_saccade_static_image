import numpy as np
import torch
import random
from pathlib import Path
from typing import Tuple, Dict, Any, List

from .photon_sim.photon_filter_func_pos_based import filtering_moving_img, filtering_static_img
from .electric_sim.electric_filter import ElectricFilter
from .electric_sim.simple_electric_filter import process_photon_signals

from ...utils import write_video



def simulation_static_rf_exp_function(
        image_array: np.ndarray,
        simulation_type: str,
        with_jitter:bool,
        shift_pixel_list:List[int],
        shift_degree_list:List[float],
        config: Dict[str, Any],
        shift_freq: float = None,
        shift_pattern: str = "square",
        sim_time_len: float = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates compound eye sampling from a rotating image to get electric signals.

    This function wraps the logic of `sim_moving_grating` and `filter_all_electric`.

    Args:
        image_array (np.ndarray): The source 2D grayscale image.
        simulation_type (str): "with" or "without" adaptation mechanism.
        config (Dict): A dictionary containing simulation parameters like 'src_img_size',
                       'omm_sqrt_num', 'img_rotate_speed'.
        saving_flag (bool): If True, saves intermediate results.
        intermediate_paths (Dict): A dictionary of paths for saving intermediates,
                                  e.g., {'src_video_path': ..., 'kernel_path': ..., 'photon_path': ...}.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
        - Rotated source image sequence (T, H, W)
        - Photon signal (T, S, S)
        - Electric signal (T_elec, S*S) where S is omm_sqrt_num
        - Used kernel data (T, K, K)
    """

    # --- 1. Image to Photon Signal (from `sim_moving_grating`) ---
    print(f"\nRunning Photon Simulation (type: {simulation_type})...")
    # `filtering_moving_img` is a black box

    photon_res, pos_rcd, shift_pix_float_rcd, rotated_source_sequence, kernel_data = filtering_static_img(
        img_array=image_array,  # Modified to take an array instead of path
        img_size=config['src_img_size'],
        sim_len=sim_time_len,
        with_saccade=(simulation_type == "with"),
        omm_num_sqrt=config['omm_sqrt_num'],
        with_jitter=with_jitter,
        shift_pixel_list=shift_pixel_list,
        shift_degree_list=shift_degree_list,
        shift_freq=shift_freq,
        shift_pattern=shift_pattern,
    )

    electric_res = None  # we included the photon-electric transduction and de-filtering all in the following line
    photon_res_defiltered = process_photon_signals(photon_res, sim_time_len, snr=60., debug_flag=False)
    return rotated_source_sequence, photon_res_defiltered, electric_res, kernel_data, pos_rcd, shift_pix_float_rcd

