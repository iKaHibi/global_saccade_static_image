import numpy as np
import time
import matplotlib.pyplot as plt

# --- GPU/CPU Backend Selection ---
# We will try to use cupy (NVIDIA GPU) as requested via the pycuda hint.
# If cupy is not available, we will fall back to scipy/numpy (CPU).
try:
    import cupy as cp
    from cupyx.scipy import signal as cusignal
    from cupy.fft import rfft as cu_rfft, irfft as cu_irfft

    _GPU_AVAILABLE = True
    print("CuPy (GPU) backend found. Processing will be performed on the GPU.")
except ImportError:
    _GPU_AVAILABLE = False
    print("CuPy (GPU) backend NOT found. Falling back to SciPy/NumPy (CPU).")
    # Use CPU libraries
    import scipy.signal as spsignal
    import scipy.fft as spfft

    # Make a cupy-like interface for the CPU
    cp = np
    cusignal = spsignal
    cu_rfft = spfft.rfft
    cu_irfft = spfft.irfft


def _get_gamma_impulse_cpu(n, tau, fs, num_samples):
    """
    Generates a normalized gamma filter impulse response on the CPU.
    """
    t = np.arange(num_samples) / fs

    # Gamma function: (t^(n-1)) * exp(-t/tau)
    impulse = (t ** (n - 1)) * np.exp(-t / tau)

    # Set t=0 to 0 (to avoid 0^n-1 issues if n < 1)
    impulse[0] = 0

    # Normalize the impulse response so its sum is 1 (preserves DC component)
    impulse_sum = np.sum(impulse)
    if impulse_sum > 0:
        impulse /= impulse_sum

    return impulse.astype(np.float32)


def _process_trials(photon_res_backend, H_gamma_backend, snr_linear, orig_samples, new_samples, backend_is_gpu=False):
    """
    Core processing pipeline that runs on either CPU (numpy) or GPU (cupy).
    """

    # --- 1. Interpolate from 100Hz to 1000Hz ---
    # We use resample, which is an FFT-based sinc interpolation
    # This is *critical* to avoid aliasing the fast gamma filter.
    if backend_is_gpu:
        # cupyx.scipy.signal.resample
        upsampled_signal = cp.repeat(photon_res_backend, 10, axis=1)
    else:
        # scipy.signal.resample
        upsampled_signal = cp.repeat(photon_res_backend, 10, axis=1)

    # --- 2. Apply Gamma Filter (Convolution) ---
    # We perform convolution in the frequency domain (FFT -> multiply -> IFFT)
    # as it's much faster for many parallel trials.
    X = cu_rfft(upsampled_signal, axis=1)
    Y_filtered = X * H_gamma_backend
    filtered_signal = cu_irfft(Y_filtered, axis=1)

    # --- 3. Add Noise ---
    # Calculate signal power for each trial
    signal_power = cp.mean(filtered_signal ** 2, axis=1, keepdims=True)

    # Calculate noise power from linear SNR
    noise_power = signal_power / snr_linear

    # Calculate noise standard deviation
    noise_std = cp.sqrt(noise_power)

    # Generate and add noise
    # Using float32 is fine for signals and saves GPU memory
    noise = cp.random.normal(0.0, 1.0, filtered_signal.shape, dtype=cp.float32)
    noise *= noise_std  # Scale noise to the correct std dev

    noisy_signal = filtered_signal + noise

    # --- 4. De-filter (Wiener Deconvolution) ---
    # This is the crucial step. We use a regularized inverse filter
    # H_wiener = H* / (|H|^2 + 1/SNR_linear)

    H_abs_sq = cp.abs(H_gamma_backend) ** 2
    # The regularization term (1/SNR) is the Noise-to-Signal ratio
    regularization = 1.0 / snr_linear
    H_wiener = cp.conj(H_gamma_backend) / (H_abs_sq + regularization)

    # Apply the Wiener filter
    Y_noisy = cu_rfft(noisy_signal, axis=1)
    X_deconvolved = Y_noisy * H_wiener
    deconvolved_signal = cu_irfft(X_deconvolved, axis=1)

    # --- 5. Downsample back to 100Hz ---
    if backend_is_gpu:
        defiltered_downsampled = cusignal.resample(deconvolved_signal, orig_samples, axis=1)
    else:
        defiltered_downsampled = cusignal.resample(deconvolved_signal, orig_samples, axis=1)

    # --- Debug Data (Optional) ---
    # Return intermediate signals for plotting
    debug_data = {
        "upsampled": upsampled_signal,
        "filtered": filtered_signal,
        "noisy": noisy_signal,
        "deconvolved": deconvolved_signal
    }

    return defiltered_downsampled, debug_data


def process_photon_signals(photon_res, sim_len, snr, debug_flag=False):
    """
    Processes photon response trials using gamma filtering, noise addition,
    and Wiener deconvolution, with GPU acceleration if available.

    Args:
        photon_res (np.ndarray): Input signals. Shape (num_trials, 300) OR (300, num_trials).
        snr (float): Signal-to-Noise Ratio in decibels (dB). E.g., 20 (good) or 5 (bad).
        debug_flag (bool): If True, plots a comparison for one random trial.

    Returns:
        np.ndarray: The de-filtered photon response signals, in the same
                    shape as the input.
    """
    # print(f"Starting processing for {photon_res.shape} array with SNR={snr}dB.")

    # --- 0. Handle Input Shape Ambiguity ---
    # The user's description "shape (300, 32400) representing 32400 trials"
    # implies the shape is (samples, trials).
    # Our processing is faster if (trials, samples).
    orig_shape = photon_res.shape
    transposed = False

    # We assume 500 is the number of samples (5s @ 100Hz)
    sim_steps = int(sim_len * 100)
    if orig_shape[0] == sim_steps and orig_shape[1] > sim_steps:
        # print(f"Input shape is ({sim_steps}, {orig_shape[1]}). Assuming (samples, trials).")
        # print("Transposing to (trials, samples) for processing.")
        photon_res = photon_res.T
        transposed = True
    elif orig_shape[1] == sim_steps and orig_shape[0] > sim_steps:
        # This is the (trials, samples) case, which is what we want.
        pass
    elif orig_shape[1] != sim_steps:
        print(f"Warning: Expected {sim_steps} samples ({sim_len}s @ 100Hz), but got {orig_shape[1]} samples.")

    n_trials, orig_samples = photon_res.shape
    # print(f"Processing {n_trials} trials of {orig_samples} samples each.")

    # --- 1. Define Parameters ---
    orig_fs = 100.0  # Hz
    new_fs = 1000.0  # Hz (Interpolation target)
    duration = orig_samples / orig_fs
    new_samples = int(duration * new_fs)

    # Gamma filter params
    gamma_n = 8
    gamma_tau = 0.003  # 3 ms

    # SNR linear scale
    snr_linear = 10.0 ** (snr / 10.0)

    # --- 2. Prepare Filter ---
    # Generate impulse on CPU (it's small and fast)
    gamma_impulse_cpu = _get_gamma_impulse_cpu(gamma_n, gamma_tau, new_fs, new_samples)

    # Get filter frequency response (FFT)
    # We calculate the FFT length needed for frequency-domain convolution
    # For speed, we just pad to the signal length.
    if _GPU_AVAILABLE:
        # Move impulse to GPU
        gamma_impulse_gpu = cp.asarray(gamma_impulse_cpu)
        # Calculate filter FFT on GPU
        H_gamma_backend = cu_rfft(gamma_impulse_gpu, n=new_samples)
        # Move signal data to GPU
        photon_res_backend = cp.asarray(photon_res, dtype=cp.float32)
    else:
        # Calculate filter FFT on CPU
        H_gamma_backend = cu_rfft(gamma_impulse_cpu, n=new_samples)
        # Keep signal data on CPU
        photon_res_backend = photon_res.astype(np.float32)

    # --- 3. Run Core Processing ---
    start_time = time.time()

    defiltered_res_backend, debug_data = _process_trials(
        photon_res_backend,
        H_gamma_backend,
        snr_linear,
        orig_samples,
        new_samples,
        backend_is_gpu=_GPU_AVAILABLE
    )

    # Wait for GPU to finish if applicable
    if _GPU_AVAILABLE:
        cp.cuda.Stream.null.synchronize()

    end_time = time.time()
    # print(f"Processing finished in {end_time - start_time:.4f} seconds.")

    # --- 4. Move Data back to CPU (if on GPU) ---
    if _GPU_AVAILABLE:
        defiltered_photon_res = cp.asnumpy(defiltered_res_backend)
    else:
        defiltered_photon_res = defiltered_res_backend

    # --- 5. Debug Plot ---
    if debug_flag:
        print("Generating debug plot for one random trial...")
        trial_idx = np.random.randint(0, n_trials)

        # Get original signal (CPU)
        original_signal = photon_res[trial_idx, :]

        # Get intermediate signals (move from GPU if needed)
        if _GPU_AVAILABLE:
            upsampled_cpu = debug_data["upsampled"][trial_idx].get()
            noisy_cpu = debug_data["noisy"][trial_idx].get()
            deconvolved_cpu = debug_data["deconvolved"][trial_idx].get()
        else:
            upsampled_cpu = debug_data["upsampled"][trial_idx]
            noisy_cpu = debug_data["noisy"][trial_idx]
            deconvolved_cpu = debug_data["deconvolved"][trial_idx]

        # Get final signal (CPU)
        deconvolved_final_cpu = defiltered_photon_res[trial_idx]

        # Time axes
        t_orig = np.linspace(0, duration, orig_samples)
        t_new = np.linspace(0, duration, new_samples)

        plt.figure(figsize=(16, 12))

        # --- Plot 1: Original vs Upsampled ---
        plt.subplot(3, 1, 1)
        plt.plot(t_orig, original_signal, 'bo-', markersize=3, label='Original Signal (100Hz)')
        plt.plot(t_new, upsampled_cpu, 'k-', alpha=0.6, label='Upsampled Signal (1000Hz)')
        plt.title(f'Trial {trial_idx}: 1. Original and Upsampled Signal')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)

        # --- Plot 2: Noisy vs Deconvolved (High Res) ---
        plt.subplot(3, 1, 2)
        plt.plot(t_new, noisy_cpu, 'r-', alpha=0.5, label=f'Filtered + Noise (SNR={snr}dB)')
        plt.plot(t_new, deconvolved_cpu, 'g--', label='Deconvolved (1000Hz)')
        plt.title('2. Noisy Signal vs. Wiener Deconvolution (at 1000Hz)')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)

        # --- Plot 3: Original vs Final Result ---
        plt.subplot(3, 1, 3)
        plt.plot(t_orig, original_signal, 'k-', alpha=0.7, label='Original Signal (100Hz)')
        plt.plot(t_orig, deconvolved_final_cpu, 'r--', label='Final Deconvolved (100Hz)')

        # Calculate and show error
        rms_error = np.sqrt(np.mean((original_signal - deconvolved_final_cpu) ** 2))
        plt.title(f'3. Original vs. Final Deconvolved Result (RMS Error: {rms_error:.4f})')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        plt.show()

    # --- 6. Return in Original Shape ---
    if transposed:
        # print("Transposing result back to original (samples, trials) shape.")
        return defiltered_photon_res.T
    else:
        return defiltered_photon_res


# --- Main execution block to test the function ---
if __name__ == "__main__":
    # --- 1. Create Mock Data ---
    N_TRIALS = 32400
    N_SAMPLES_ORIG = 300  # 3 seconds @ 100Hz
    FS_ORIG = 100.0
    DURATION = N_SAMPLES_ORIG / FS_ORIG

    print(f"Generating mock data: {N_TRIALS} trials, {N_SAMPLES_GUIDE, {N_SAMPLES_ORIG}} samples...")

    t = np.linspace(0, DURATION, N_SAMPLES_ORIG, endpoint=False)
    # Create a sum of sines as a test signal
    s1 = np.sin(2 * np.pi * 2 * t)  # 2 Hz
    s2 = np.sin(2 * np.pi * 5 * t)  # 5 Hz
    s3 = np.sin(2 * np.pi * 15 * t)  # 15 Hz

    # Create 32400 trials by adding some phase-shifted sines
    trial_indices = np.arange(N_TRIALS)[:, np.newaxis]
    t_batch = t[np.newaxis, :]

    # Mock data with some variability
    mock_photon_res = (
            np.sin(2 * np.pi * 2 * t_batch + trial_indices * 0.01) +
            0.5 * np.sin(2 * np.pi * 8 * t_batch + trial_indices * 0.05) +
            0.2 * np.random.randn(N_TRIALS, N_SAMPLES_ORIG)  # A little white noise
    )

    # Normalize
    mock_photon_res = (mock_photon_res.T - mock_photon_res.mean(axis=1)).T
    mock_photon_res = (mock_photon_res.T / mock_photon_res.std(axis=1)).T

    print(f"Mock data shape: {mock_photon_res.shape}")

    # --- 2. Test with a "bad" SNR ---
    # With low SNR, the de-filtered result will be heavily smoothed
    # (because the Wiener filter trusts the noisy data less)
    print("\n--- TEST 1: Low SNR (5 dB) ---")
    defiltered_res_low_snr = process_photon_signals(
        mock_photon_res,
        snr=5.0,
        debug_flag=True
    )

    # --- 3. Test with a "good" SNR ---
    # With high SNR, the de-filtered result should be very close
    # to the original signal.
    print("\n--- TEST 2: High SNR (25 dB) ---")
    defiltered_res_high_snr = process_photon_signals(
        mock_photon_res,
        snr=25.0,
        debug_flag=True
    )

    # --- 4. Test Transpose Logic ---
    print("\n--- TEST 3: Transposed Input (25 dB) ---")
    defiltered_res_transposed = process_photon_signals(
        mock_photon_res.T,  # Pass in (300, 32400)
        snr=25.0,
        debug_flag=False  # No plot this time
    )
    print(f"Original input shape: {mock_photon_res.T.shape}")
    print(f"Final output shape: {defiltered_res_transposed.shape}")
    # Check if the result is numerically close to the non-transposed run
    assert np.allclose(defiltered_res_high_snr, defiltered_res_transposed.T)
    print("Transpose test PASSED.")

