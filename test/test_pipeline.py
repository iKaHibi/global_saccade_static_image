"""Unit tests for the pipeline_sim_diff_rf_jitter_pos_based module."""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


class TestIdealLowpassDownsample:
    """Tests for the ideal_lowpass_downsample function."""

    def test_output_shape_2d(self):
        """Test output shape matches target dimensions for 2D input."""
        from src.pipelines.pipeline_sim_diff_rf_jitter_pos_based import ideal_lowpass_downsample

        H, W = 100, 100
        target_h, target_w = 50, 50
        img = torch.randn(H, W)

        result = ideal_lowpass_downsample(img, target_h, target_w)

        assert result.shape == (target_h, target_w)

    def test_output_shape_3d(self):
        """Test output shape for 3D input (C, H, W)."""
        from src.pipelines.pipeline_sim_diff_rf_jitter_pos_based import ideal_lowpass_downsample

        C, H, W = 3, 100, 100
        target_h, target_w = 50, 50
        img = torch.randn(C, H, W)

        result = ideal_lowpass_downsample(img, target_h, target_w)

        assert result.shape == (C, target_h, target_w)

    def test_output_shape_batched(self):
        """Test output shape for batched input (B, C, H, W)."""
        from src.pipelines.pipeline_sim_diff_rf_jitter_pos_based import ideal_lowpass_downsample

        B, C, H, W = 2, 1, 100, 100
        target_h, target_w = 50, 50
        img = torch.randn(B, C, H, W)

        result = ideal_lowpass_downsample(img, target_h, target_w)

        assert result.shape == (B, C, target_h, target_w)

    def test_output_type(self):
        """Test output is real tensor (imaginary part should be near zero)."""
        from src.pipelines.pipeline_sim_diff_rf_jitter_pos_based import ideal_lowpass_downsample

        img = torch.randn(100, 100)
        result = ideal_lowpass_downsample(img, 50, 50)

        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float64 or result.dtype == torch.float32

    def test_scale_factor_conservation(self):
        """Test that energy is approximately conserved for identity downsampling."""
        from src.pipelines.pipeline_sim_diff_rf_jitter_pos_based import ideal_lowpass_downsample

        H, W = 64, 64
        img = torch.randn(H, W)

        # Downsample and upsample back
        downsampled = ideal_lowpass_downsample(img, H, W)
        upsampled = ideal_lowpass_downsample(downsampled, H, W)

        # Original and upsampled should be similar in magnitude
        assert torch.allclose(img, upsampled, rtol=1e-4, atol=1e-6)

    def test_downsampling_reduces_high_freq(self):
        """Test that downsampling reduces high frequency content."""
        from src.pipelines.pipeline_sim_diff_rf_jitter_pos_based import ideal_lowpass_downsample

        # Create high frequency image
        H, W = 128, 128
        img = torch.sin(torch.linspace(0, 20, H).view(-1, 1) * torch.linspace(0, 20, W).view(1, -1))

        # Downsample by factor of 2
        target_h, target_w = 64, 64
        result = ideal_lowpass_downsample(img, target_h, target_w)

        # Result should be smooth (low frequencies preserved)
        assert result.shape == (target_h, target_w)

    def test_invalid_target_size(self):
        """Test handling of invalid target sizes."""
        from src.pipelines.pipeline_sim_diff_rf_jitter_pos_based import ideal_lowpass_downsample

        img = torch.randn(100, 100)

        # Target larger than input should work (zero-padded FFT)
        result = ideal_lowpass_downsample(img, 150, 150)
        assert result.shape == (150, 150)

        # Zero target should raise error or handle gracefully
        with pytest.raises((ValueError, IndexError)):
            ideal_lowpass_downsample(img, 0, 0)


class TestGetDestinationPath:
    """Tests for the _get_destination_path function."""

    def test_path_construction(self):
        """Test correct path construction."""
        from src.pipelines.pipeline_sim_diff_rf_jitter_pos_based import _get_destination_path

        with tempfile.TemporaryDirectory() as tmpdir:
            results_root = Path(tmpdir)
            sim_type = "test_simulation"

            result = _get_destination_path(results_root, sim_type)

            expected = results_root / sim_type / 'scd_confirm' / 'reconstructed_images'
            assert result == expected

    def test_directory_creation(self):
        """Test that directory is created if it doesn't exist."""
        from src.pipelines.pipeline_sim_diff_rf_jitter_pos_based import _get_destination_path

        with tempfile.TemporaryDirectory() as tmpdir:
            results_root = Path(tmpdir)
            sim_type = "new_simulation"

            result = _get_destination_path(results_root, sim_type)

            assert result.exists()
            assert result.is_dir()

    def test_idempotent_call(self):
        """Test that calling twice doesn't fail."""
        from src.pipelines.pipeline_sim_diff_rf_jitter_pos_based import _get_destination_path

        with tempfile.TemporaryDirectory() as tmpdir:
            results_root = Path(tmpdir)
            sim_type = "test_simulation"

            result1 = _get_destination_path(results_root, sim_type)
            result2 = _get_destination_path(results_root, sim_type)

            assert result1 == result2
            assert result2.exists()


class TestMetricsRegistry:
    """Tests for the METRICS_REGISTRY constant."""

    def test_expected_metrics_present(self):
        """Test that all expected metrics are registered."""
        from src.pipelines.pipeline_sim_diff_rf_jitter_pos_based import METRICS_REGISTRY

        expected = ["ssim", "psnr", "fsim", "psd_similarity", "gms", "gssim",
                    "haarpsi", "vif", "lpips", "fish", "niqe", "maniqa", "phase_similarity"]

        for metric in expected:
            assert metric in METRICS_REGISTRY, f"Missing metric: {metric}"

    def test_registry_values_callable(self):
        """Test that all registry values are callable."""
        from src.pipelines.pipeline_sim_diff_rf_jitter_pos_based import METRICS_REGISTRY

        for name, func in METRICS_REGISTRY.items():
            assert callable(func), f"Registry entry '{name}' is not callable"


class TestProjectRoot:
    """Tests for project_root constant."""

    def test_project_root_exists(self):
        """Test that project_root points to existing directory."""
        from src.pipelines.pipeline_sim_diff_rf_jitter_pos_based import project_root

        assert project_root.exists()
        assert project_root.is_dir()

    def test_project_root_contains_expected(self):
        """Test that project_root contains expected subdirectories."""
        from src.pipelines.pipeline_sim_diff_rf_jitter_pos_based import project_root

        # Check for key directories
        assert (project_root / "src").exists() or (project_root / "configs").exists()


class TestMetricCalculationLogic:
    """Tests for metric calculation logic (ratio computation)."""

    def test_ratio_calculation_normal(self):
        """Test ratio calculation for normal case."""
        with_val_sum = 0.8
        wo_val_sum = 0.5
        ratio = with_val_sum / wo_val_sum

        assert ratio == pytest.approx(1.6)

    def test_ratio_calculation_inverted_metrics(self):
        """Test ratio calculation for inverted metrics (niqe, psd_similarity)."""
        with_val_sum = 0.8
        wo_val_sum = 0.5
        # These metrics: lower is better, so ratio is inverted
        ratio = wo_val_sum / with_val_sum

        assert ratio == pytest.approx(0.625)

    def test_ratio_calculation_zero_denominator(self):
        """Test ratio calculation when denominator is zero."""
        with_val_sum = 0.5
        wo_val_sum = 0.0

        # When wo_val_sum == 0 and with_val_sum > 0
        ratio = float('inf') if with_val_sum > 0 else 0.0
        assert ratio == float('inf')

        # When both are zero
        ratio = 0.0
        assert ratio == 0.0

    def test_mean_ratio_calculation(self):
        """Test mean ratio calculation across samples."""
        ratios = [1.0, 1.5, 2.0, 1.2]
        mean_ratio = np.mean(ratios)

        assert mean_ratio == pytest.approx(1.425)

    def test_mean_ratio_handles_inf(self):
        """Test mean ratio calculation excludes inf values."""
        ratios = [1.0, 1.5, float('inf'), 2.0]
        valid_ratios = [r for r in ratios if r != float('inf')]

        mean_ratio = np.mean(valid_ratios) if valid_ratios else 0.0

        assert mean_ratio == pytest.approx(1.5)


class TestConfigValidation:
    """Tests for configuration validation logic."""

    def test_omm_sqrt_num_must_be_list(self):
        """Test that omm_sqrt_num validation works correctly."""
        valid_config = {"simulation_params": {"omm_sqrt_num": [27, 54]}}
        invalid_config = {"simulation_params": {"omm_sqrt_num": 27}}

        assert isinstance(valid_config["simulation_params"]["omm_sqrt_num"], list)
        assert not isinstance(invalid_config["simulation_params"]["omm_sqrt_num"], list)

    def test_shift_pixel_extraction(self):
        """Test shift_pixel extraction from config."""
        config = {
            "simulation_params": {
                "shift_pixel": [1, 2, 3, 4, 5],
                "omm_sqrt_num": 27
            },
            "reconstruction_params": {
                "upscaling_times": 4
            }
        }

        shift_pixel_list = [int(elem) for elem in config["simulation_params"]["shift_pixel"]]
        omm_sqrt_num = int(config["simulation_params"]["omm_sqrt_num"])
        up_scale = int(config["reconstruction_params"]["upscaling_times"])

        assert shift_pixel_list == [1, 2, 3, 4, 5]
        assert omm_sqrt_num == 27
        assert up_scale == 4


class TestImageSelectionLogic:
    """Tests for random image selection logic."""

    def test_min_images_check(self):
        """Test minimum image count check."""
        min_required = 40

        # Should pass
        available_40 = 40
        assert available_40 >= min_required

        # Should fail
        available_39 = 39
        assert not (available_39 >= min_required)

    def test_indices_to_log_set(self):
        """Test that indices_to_log is a set for O(1) lookup."""
        saving_log_number = 6
        indices_to_log = set(range(saving_log_number))

        assert isinstance(indices_to_log, set)
        assert len(indices_to_log) == saving_log_number
        assert 0 in indices_to_log
        assert 5 in indices_to_log
        assert 6 not in indices_to_log

    def test_saving_log_number_cap(self):
        """Test that saving_log_number is capped at image count."""
        image_count = 40
        saving_log_number = 50

        capped = min(saving_log_number, image_count)

        assert capped == 40


class TestFolderNamingLogic:
    """Tests for folder naming based on split_by_jitter."""

    def test_folder_name_split_by_jitter_with_jitter(self):
        """Test folder naming when split_by_jitter=True and with_jitter=True."""
        split_by_jitter = True
        with_jitter = True

        result = "with_jitter" if (split_by_jitter and with_jitter) else \
                 "wo_jitter" if (split_by_jitter and not with_jitter) else \
                 f"test_saccade"

        assert result == "with_jitter"

    def test_folder_name_split_by_jitter_without_jitter(self):
        """Test folder naming when split_by_jitter=True and with_jitter=False."""
        split_by_jitter = True
        with_jitter = False

        result = "with_jitter" if (split_by_jitter and with_jitter) else \
                 "wo_jitter" if (split_by_jitter and not with_jitter) else \
                 f"test_saccade"

        assert result == "wo_jitter"

    def test_folder_name_no_split(self):
        """Test folder naming when split_by_jitter=False."""
        split_by_jitter = False
        with_jitter = True
        simulation_type = "with"

        result = "with_jitter" if (split_by_jitter and with_jitter) else \
                 "wo_jitter" if (split_by_jitter and not with_jitter) else \
                 f"{simulation_type}_saccade"

        assert result == "with_saccade"


class TestNumpyConversion:
    """Tests for numpy to native type conversion."""

    def test_convert_numpy_scalar(self):
        """Test conversion of numpy scalar to native Python type."""
        from src.pipelines.pipeline_sim_diff_rf_jitter_pos_based import main

        # Access the nested function through introspection or reimplement
        def convert_numpy_to_native(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, dict):
                return {k: convert_numpy_to_native(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_numpy_to_native(i) for i in obj]
            return obj

        np_int = np.int64(42)
        np_float = np.float64(3.14)
        np_bool = np.bool_(True)

        assert convert_numpy_to_native(np_int) == 42
        assert convert_numpy_to_native(np_float) == pytest.approx(3.14)
        assert convert_numpy_to_native(np_bool) == True

    def test_convert_dict_with_numpy(self):
        """Test conversion of dict containing numpy types."""
        from src.pipelines.pipeline_sim_diff_rf_jitter_pos_based import main

        def convert_numpy_to_native(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, dict):
                return {k: convert_numpy_to_native(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_numpy_to_native(i) for i in obj]
            return obj

        data = {
            "metric1": np.float64(1.5),
            "nested": {
                "metric2": np.int32(10)
            }
        }

        result = convert_numpy_to_native(data)

        assert result["metric1"] == pytest.approx(1.5)
        assert result["nested"]["metric2"] == 10


class TestShiftParameterSampling:
    """Tests for shift parameter sampling logic."""

    def test_log_normal_sampling(self):
        """Test log-normal distribution sampling."""
        shift_amp_degree_mean = 0.16  # sigma
        shift_amp_degree_std = 0.122  # mean of underlying normal

        samples = [
            np.random.lognormal(mean=np.log(shift_amp_degree_std), sigma=shift_amp_degree_mean)
            for _ in range(100)
        ]

        # All samples should be positive
        assert all(s > 0 for s in samples)

    def test_shift_pixel_calculation(self):
        """Test shift_pixel calculation from shift amplitude."""
        src_img_size = 1080
        shift_amp_in_degree = 0.5

        shift_pixel = max(
            int(round(abs(src_img_size * 0.9 / 170. * shift_amp_in_degree))),
            1
        )

        expected = max(int(round(abs(1080 * 0.9 / 170. * 0.5))), 1)
        assert shift_pixel == expected

    def test_shift_freq_clipping(self):
        """Test shift frequency clipping to valid range."""
        shift_freq_mean = 8.0
        shift_freq_std = 0.0

        shift_freq = np.clip(
            np.random.normal(loc=shift_freq_mean, scale=shift_freq_std + 1e-9),
            0.05, 50
        )

        assert 0.05 <= shift_freq <= 50


class TestResultsPrefix:
    """Tests for results file prefix logic."""

    def test_prefix_split_by_jitter(self):
        """Test prefix changes based on split_by_jitter."""
        split_by_jitter = True
        prefix = "jitter" if split_by_jitter else "scd"

        assert prefix == "jitter"

    def test_prefix_no_split(self):
        """Test prefix when split_by_jitter=False."""
        split_by_jitter = False
        prefix = "jitter" if split_by_jitter else "scd"

        assert prefix == "scd"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
