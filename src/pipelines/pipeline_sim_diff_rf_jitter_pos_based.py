import copy
import json
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import torch
import torch.fft
from PIL import Image
from tqdm import tqdm

from .. import utils
from src.apis.compound_eye_utils.e2p_decoding.e2p_lstm import LSTMDecoder
from src.apis.compound_eye_utils.reconstruction_static_api_pos_based import reconstruction_with_jitter_function
from src.apis.compound_eye_utils.simulation_api_pos_based import simulation_static_rf_exp_function
from src.apis.metric_cal_utils.fsim_api import fsim_api_torch
from src.apis.metric_cal_utils.gsm_api import gms_api_torch
from src.apis.metric_cal_utils.gssim_api import gssim_api_torch
from src.apis.metric_cal_utils.haarips_api import haarpsi_api_torch
from src.apis.metric_cal_utils.lpips_api import lpips_api_torch
from src.apis.metric_cal_utils.maniqa_api import calculate_maniqa
from src.apis.metric_cal_utils.niqe_api import calculate_niqe
from src.apis.metric_cal_utils.psd_similarity import calculate_phase_similarity, calculate_spectral_metrics
from src.apis.metric_cal_utils.psnr_api import psnr_api_torch
from src.apis.metric_cal_utils.sharpness_api import calculate_fish_on_tensor
from src.apis.metric_cal_utils.ssim_custom_api import ssim_api_torch
from src.apis.metric_cal_utils.vif_api import vif_api_torch

METRICS_REGISTRY: Dict[str, Callable] = {
    "ssim": ssim_api_torch,
    "psnr": psnr_api_torch,
    "fsim": fsim_api_torch,
    "psd_similarity": calculate_spectral_metrics,
    "gms": gms_api_torch,
    "gssim": gssim_api_torch,
    "haarpsi": haarpsi_api_torch,
    "vif": vif_api_torch,
    "lpips": lpips_api_torch,
    "fish": calculate_fish_on_tensor,
    "niqe": calculate_niqe,
    "maniqa": calculate_maniqa,
    "phase_similarity": calculate_phase_similarity,
}

project_root = Path(__file__).resolve().parent.parent.parent.parent


def ideal_lowpass_downsample(img: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Downsample using Ideal Low-pass Filter (Sinc Interpolation) via DFT cropping."""
    *batch_dims, H, W = img.shape

    f_img = torch.fft.fft2(img, dim=(-2, -1))
    f_shift = torch.fft.fftshift(f_img, dim=(-2, -1))

    c_h, c_w = H // 2, W // 2
    start_h = c_h - target_h // 2
    start_w = c_w - target_w // 2
    f_crop = f_shift[..., start_h:start_h + target_h, start_w:start_w + target_w]

    f_crop_ishift = torch.fft.ifftshift(f_crop, dim=(-2, -1))
    img_down = torch.fft.ifft2(f_crop_ishift, dim=(-2, -1))

    scale_factor = (target_h * target_w) / (H * W)
    return img_down.real * scale_factor


def run_comparison(
        preprocessed_folder: Path,
        simulation_type: str,
        with_jitter: bool,
        saving_log_number: int,
        config: Dict[str, Any],
        device: str,
        metrics_to_estimate: List[str],
        split_by_jitter: bool = False,
        shift_degree_list: List[float] = [0.],
        shift_freq: float = None,
) -> Dict[str, Any]:
    """Run full simulation and reconstruction pipeline."""
    print(f"\n--- Starting Comparison Stage: simulation_type='{simulation_type}', with_jitter='{with_jitter}' ---")

    sim_params = config["simulation_params"]
    recon_params = config["reconstruction_params"]

    shift_pixel_list = [int(elem) for elem in sim_params["shift_pixel"]]
    omm_sqrt_num = int(sim_params["omm_sqrt_num"])
    up_scale = int(recon_params["upscaling_times"])
    sim_time_len = float(sim_params["sim_time_len"])
    shift_pattern = sim_params["shift_pattern"]

    results_root = Path(
        config.get('data_folders', {}).get('results', 'results')
    ) / "d_test" / f"omm_sqrt_num_{omm_sqrt_num}_pos_based_1_7degree"

    # Randomly select 40 images
    all_images = list(preprocessed_folder.glob('*.png'))
    if len(all_images) < 40:
        raise FileNotFoundError(f"Need at least 40 images, found {len(all_images)}")
    image_paths = random.sample(all_images, 40)

    saving_log_number = min(saving_log_number, len(image_paths))
    indices_to_log = set(range(saving_log_number))
    print(f"Randomly selected {saving_log_number} images to log intermediate data for.")

    image_names = []
    results = {metric_name: {} for metric_name in metrics_to_estimate}

    # Prepare e2p model
    e2p_pre_trained_pth = os.path.join(project_root, config["model_paths"]["e2p_model_path"])
    e2p_model = LSTMDecoder(64, num_layers=6)
    e2p_model.to(device)

    for i, img_path in enumerate(
            tqdm(image_paths, desc=f"Processing Images ({simulation_type}, jitter: {with_jitter})", position=0, leave=True)):
        image_names.append(img_path.stem)
        saving_flag = i in indices_to_log
        image_array = np.array(Image.open(img_path))

        rotated_video, photon_signal, electric_signal, kernels, pos_rcd, shift_pix_float_rcd = \
            simulation_static_rf_exp_function(
                image_array=image_array,
                simulation_type=simulation_type,
                with_jitter=with_jitter,
                shift_pixel_list=shift_pixel_list,
                shift_degree_list=shift_degree_list,
                config=sim_params,
                shift_freq=shift_freq,
                shift_pattern=shift_pattern,
                sim_time_len=sim_time_len
            )

        sub_folder_str = "with_jitter" if (split_by_jitter and with_jitter) else \
                         "wo_jitter" if (split_by_jitter and not with_jitter) else \
                         f"{simulation_type}_saccade"
        destination_dir = _get_destination_path(results_root, sub_folder_str)

        ground_truth_frame, reconstructed_frame = reconstruction_with_jitter_function(
            rotated_source_sequence=rotated_video,
            electric_signal=electric_signal,
            pos_rcd=pos_rcd,
            shift_pix_float_rcd=shift_pix_float_rcd,
            kernel_data=kernels,
            e2p_model=e2p_model,
            omm_num_sqrt=sim_params["omm_sqrt_num"],
            shift_pixel_list=shift_pixel_list,
            device=device,
            config=recon_params,
            saving_flag=saving_flag,
            destination_path=destination_dir if saving_flag else None,
            destination_file_name=img_path.stem if saving_flag else None,
            with_jitter=with_jitter,
            src_img_size=sim_params["src_img_size"],
            up_scale=up_scale,
            photon_signal=photon_signal / 255
        )

        for metric_name in metrics_to_estimate:
            if metric_name not in METRICS_REGISTRY:
                print(f"Warning: Feature '{metric_name}' not registered. Skipping.")
                continue

            api_function = METRICS_REGISTRY[metric_name]
            recon_frame_tensor = torch.tensor(reconstructed_frame, dtype=torch.float32) / 255.
            ground_truth_frame_tensor = torch.tensor(ground_truth_frame, dtype=torch.float32) / 255.
            slice_len = int(round(ground_truth_frame_tensor.shape[-2] * 0.05))
            ground_truth_frame_tensor = ground_truth_frame_tensor[slice_len:-slice_len, slice_len:-slice_len]

            if metric_name in ("fish", "niqe", "maniqa"):
                metric_value = api_function(recon_frame_tensor)
            else:
                metric_value = api_function(recon_frame_tensor.unsqueeze(0), ground_truth_frame_tensor.unsqueeze(0))

            results[metric_name][img_path.stem] = [metric_value]

    results["image_names"] = image_names
    return results


def _get_destination_path(results_root: Path, sim_type: str) -> Path:
    """Construct path for saving reconstructed images."""
    dest_path = results_root / sim_type / 'scd_confirm' / 'reconstructed_images'
    dest_path.mkdir(parents=True, exist_ok=True)
    return dest_path


def main(cfg_path_str, n_samples=5, split_by_jitter=False):
    """Main entry point for the simulation pipeline."""
    config = utils.load_json(Path(cfg_path_str))

    omm_sqrt_num_list = config["simulation_params"]["omm_sqrt_num"]
    if not isinstance(omm_sqrt_num_list, list):
        raise TypeError("omm_sqrt_num must be a list")

    metrics = ["fsim", "haarpsi", "psnr", "ssim", "niqe", "fish", "psd_similarity", "phase_similarity"]

    final_aggregated_results = {}
    final_aggregated_results_all = {}

    shift_amp_degree_mean = config["simulation_params"]["shift_degree_mean"]
    shift_amp_degree_std = config["simulation_params"]["shift_degree_std"]
    shift_freq_mean = config["simulation_params"]["shift_freq_mean"]
    shift_freq_std = config["simulation_params"]["shift_freq_std"]

    for omm_sqrt_num in omm_sqrt_num_list:
        print(f"\n{'=' * 20} Running for omm_sqrt_num = {omm_sqrt_num} (n_samples={n_samples}) {'=' * 20}")

        all_samples_results = {
            "shift_amp_degree": [],
            "shift_freq": [],
            "shift_pixel": [],
            "samples": []
        }
        all_samples_results_detailed = []
        image_names = None

        for i in range(n_samples):
            print(f"--- Sample {i + 1}/{n_samples} ---")

            dist_type = config.get("simulation_params", {}).get("dist_type", "gaussian")

            if dist_type == "log_norm":
                shift_amp_in_degree_list = [
                    np.random.lognormal(mean=np.log(shift_amp_degree_std), sigma=shift_amp_degree_mean)
                    for _ in range(10)
                ]
                shift_isi = np.clip(
                    np.random.lognormal(mean=np.log(shift_freq_std + 1e-9), sigma=shift_freq_mean),
                    0.05, 50
                )
                shift_freq = 1. / shift_isi
            else:
                shift_amp_in_degree_list = [
                    np.random.normal(loc=shift_amp_degree_mean, scale=shift_amp_degree_std)
                    for _ in range(10)
                ]
                shift_freq = np.clip(
                    np.random.normal(loc=shift_freq_mean, scale=shift_freq_std + 1e-9),
                    0.05, 50
                )

            if shift_freq_std == 0:
                shift_freq = shift_freq_mean

            shift_pixel_list = [
                max(int(round(abs(config["simulation_params"]["src_img_size"] * 0.9 / 170. * shift_amp_in_degree))), 1)
                for shift_amp_in_degree in shift_amp_in_degree_list
            ]

            print(f"Shift Amp (deg): {shift_amp_in_degree_list[0]:.4f}, "
                  f"Shift Freq: {shift_freq:.4f}, Shift Pixel: {shift_pixel_list[0]}")

            all_samples_results["shift_amp_degree"].append(shift_amp_in_degree_list)
            all_samples_results["shift_freq"].append(shift_freq)
            all_samples_results["shift_pixel"].append(shift_pixel_list)

            current_config = copy.deepcopy(config)
            current_config["simulation_params"]["shift_pixel"] = shift_pixel_list
            current_config["simulation_params"]["omm_sqrt_num"] = omm_sqrt_num

            sim_type_str = "without" if split_by_jitter else "with"
            result_with_jitter = run_comparison(
                Path("data/mcgill_preprocessed/"), sim_type_str, True, 6,
                current_config, "cuda", metrics, split_by_jitter=split_by_jitter,
                shift_degree_list=shift_amp_in_degree_list, shift_freq=shift_freq
            )
            result_wo_jitter = run_comparison(
                Path("data/mcgill_preprocessed/"), "without", False, 6, current_config,
                "cuda", metrics, split_by_jitter=split_by_jitter,
                shift_degree_list=shift_amp_in_degree_list, shift_freq=shift_freq
            )

            sample_ratios = {}
            image_names = result_with_jitter["image_names"]

            for metric_name in metrics:
                with_val_sum = sum(np.mean(val) for val in result_with_jitter.get(metric_name, {}).values())
                wo_val_sum = sum(np.mean(val) for val in result_wo_jitter.get(metric_name, {}).values())

                if wo_val_sum == 0:
                    ratio = float('inf') if with_val_sum > 0 else 0.0
                    print(f"Warning: Sample {i + 1}: Sum of 'without jitter' metric '{metric_name}' is zero.")
                else:
                    ratio = wo_val_sum / with_val_sum if metric_name in ("niqe", "psd_similarity") \
                        else with_val_sum / wo_val_sum

                sample_ratios[f"{metric_name}_ratio"] = ratio
                print(f"    -> Sample={i + 1}, Metric={metric_name}, Ratio = {ratio:.4f}")

            all_samples_results["samples"].append(sample_ratios)
            all_samples_results_detailed.append({
                "sample_index": i,
                "parameters": {
                    "shift_amp_degree": shift_amp_in_degree_list,
                    "shift_freq": shift_freq,
                    "shift_pixel": shift_pixel_list,
                },
                "raw_data_with_jitter": result_with_jitter,
                "raw_data_without_jitter": result_wo_jitter
            })

        all_samples_results["image_names"] = image_names

        # Calculate mean ratios
        mean_ratios = {"num_samples": n_samples}
        for metric_name in metrics:
            all_ratios = [
                sample_res[f"{metric_name}_ratio"]
                for sample_res in all_samples_results["samples"]
                if f"{metric_name}_ratio" in sample_res and sample_res[f"{metric_name}_ratio"] != float('inf')
            ]
            mean_ratio = np.mean(all_ratios) if all_ratios else 0.0
            mean_ratios[f"{metric_name}_mean_ratio"] = mean_ratio
            print(f"  -> FINAL MEAN: omm_sqrt_num={omm_sqrt_num}, Metric={metric_name}, "
                  f"Mean Improvement Ratio = {mean_ratio:.4f}")

        all_samples_results["mean_ratios"] = mean_ratios
        final_aggregated_results[omm_sqrt_num] = all_samples_results
        final_aggregated_results_all[omm_sqrt_num] = all_samples_results_detailed
        last_omm_sqrt = int(omm_sqrt_num)

    print(f"\n{'=' * 20} Final Aggregated Results {'=' * 20}")

    prefix = "jitter" if split_by_jitter else "scd"

    def convert_numpy_to_native(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {k: convert_numpy_to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_numpy_to_native(i) for i in obj]
        return obj

    output_data_summary = convert_numpy_to_native(final_aggregated_results)
    output_data_all = convert_numpy_to_native(final_aggregated_results_all)

    filename_summary = f"results/{prefix}_sn{last_omm_sqrt}_m{shift_amp_degree_mean}s{shift_amp_degree_std}_n{n_samples}_res_p40.json"
    with open(filename_summary, 'w') as f:
        json.dump(output_data_summary, f, indent=4)
    print(f"\nSummary results saved to: {filename_summary}")

    filename_all = f"results/{prefix}_sn{last_omm_sqrt}_m{shift_amp_degree_mean}s{shift_amp_degree_std}_n{n_samples}_res_all_p40.json"
    with open(filename_all, 'w') as f:
        json.dump(output_data_all, f, indent=4)
    print(f"Detailed raw results saved to: {filename_all}")


if __name__ == "__main__":
    main()
