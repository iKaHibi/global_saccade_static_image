import numpy as np
import matplotlib.pyplot as plt


def generate_shift_sequence(time_steps_num, amplitude_seq, frequency, pattern="square", sampling_rate=100,
                            force_second_down=False, debug_flag=False):
    """
    Generates a sequence of (x, y) shift offsets with distinct Active/Stay phases and variable amplitudes.

    Logic Update:
    - First 1 second (100 steps): Stay still (0,0).
    - After 1 second: Immediately start the first Active shift.

    Parameters:
    - time_steps_num: int, total number of time steps.
    - amplitude_seq: list or float, sequence of shift magnitudes (e.g., [2, 5]).
      Cycles through this list for each active period.
    - frequency: float, cycle frequency in Hz (1 cycle = 1 Active + 1 Stay).
    - pattern: str, "square" or "sin".
    - sampling_rate: int, Hz (default 100).
    - force_second_down: bool, if True, forces the 2nd shift to be 90 degrees (down). Default False.
    - debug_flag: bool, if True, plots the trajectory and statistics.

    Returns:
    - shift_array: np.ndarray of shape (time_steps_num, 2)
    """

    # --- 1. Validation & Initialization ---

    # Handle single float input for backward compatibility
    if isinstance(amplitude_seq, (int, float)):
        amplitude_seq = [float(amplitude_seq)]

    # Check minimum length (2 seconds)
    min_steps = 2 * sampling_rate
    if time_steps_num < min_steps:
        raise ValueError(
            f"Time steps ({time_steps_num}) must correspond to at least 2 seconds ({min_steps} steps) at {sampling_rate}Hz.")

    # Create Time Array
    dt = 1.0 / sampling_rate
    t = np.arange(time_steps_num) * dt

    # Define start index for shifting (1 second mark)
    start_active_idx = int(1.0 * sampling_rate)

    # --- 2. Frequency Check ---

    # Calculate remaining time available for shifts
    remaining_duration = (time_steps_num - start_active_idx) * dt

    # Duration of a single Active phase (half a cycle)
    active_phase_duration = 1.0 / (2 * frequency)

    if remaining_duration < active_phase_duration:
        print(
            f"WARNING: Frequency {frequency}Hz is too low to complete even one shift in the remaining time ({remaining_duration:.2f}s).")

    # --- 3. Generate Waveform for the Active Segment ---

    # Initialize full arrays with zeros (handles the first 1s stay implicitly)
    shift_x = np.zeros(time_steps_num)
    shift_y = np.zeros(time_steps_num)

    # We slice the time array to process only the part after 1 second
    # We subtract 1.0 so that the phase calculation starts at 0 for the first active block
    t_slice = t[start_active_idx:] - 1.0

    # Define Event Blocks (Active vs Stay) for the slice
    events_per_second = 2 * frequency
    event_indices = np.floor(t_slice * events_per_second).astype(int)

    num_events = event_indices[-1] + 1

    # We only need data for even indices (0, 2, 4...) which are "Active"
    # Odd indices (1, 3, 5...) are "Stay"
    num_active_periods = (num_events + 1) // 2

    if num_active_periods > 0:
        valid_angles = np.arange(0, 360, 45)
        active_directions = np.random.choice(valid_angles, size=num_active_periods)

        # Override logic: force specific directions for first 2 *active* periods
        # Event 0 (Active): Rightwards (0 degrees)
        # Event 2 (Active): Downwards (90 degrees) - ONLY if force_second_down is True
        # Note: Since we check length > 2s, we usually have time for these if freq isn't tiny.
        if num_active_periods >= 1:
            active_directions[0] = 0
        if force_second_down and num_active_periods >= 2:
            active_directions[1] = 90  # Corresponds to Event 2

        # Map directions to the time slice
        active_idx_map = event_indices // 2
        active_idx_map = np.clip(active_idx_map, 0, num_active_periods - 1)

        angles_sequence_deg = active_directions[active_idx_map]
        angles_sequence_rad = np.deg2rad(angles_sequence_deg)

        # Generate Amplitude Sequence map
        active_amplitudes_list = np.array([
            amplitude_seq[i % len(amplitude_seq)]
            for i in range(num_active_periods)
        ])

        # Map amplitudes to full time slice
        current_target_amplitude = active_amplitudes_list[active_idx_map]

        # Generate Unipolar Waveform
        event_progress = (t_slice * events_per_second) % 1.0

        if pattern == "square":
            base_magnitude = current_target_amplitude
        elif pattern == "sin":
            base_magnitude = current_target_amplitude * np.sin(np.pi * event_progress)
        else:
            raise ValueError("Pattern must be 'square' or 'sin'")

        # Apply "Stay" Mask to the slice (Mask magnitude to 0 for every odd event)
        is_stay_event = (event_indices % 2 == 1)
        base_magnitude[is_stay_event] = 0.0

        # Project to X and Y and assign to the main arrays
        shift_x[start_active_idx:] = base_magnitude * np.cos(angles_sequence_rad)
        shift_y[start_active_idx:] = base_magnitude * np.sin(angles_sequence_rad)

    # Stack into (N, 2) array
    shift_array = np.column_stack((shift_x, shift_y))

    # 4. Debug Visualization
    if debug_flag:
        # For visualization, we pass the directions used in the slice
        viz_angles = active_directions if num_active_periods > 0 else []
        _visualize_shifts(shift_x, shift_y, t, viz_angles, pattern)

    return shift_array


def _visualize_shifts(x, y, t, unique_angles, pattern):
    """Helper function to visualize the generated data."""

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(f"Shift Generation Debug: {pattern.title()} Wave (Variable Amplitudes)", fontsize=16)

    # Plot 1: Trajectory (2D Plane)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(x, y, alpha=0.6, linewidth=1)
    ax1.scatter(x[0], y[0], color='green', label='Start')
    ax1.scatter(x[-1], y[-1], color='red', label='End')
    ax1.set_title("2D Trajectory (X vs Y)")
    ax1.set_xlabel("X Offset")
    ax1.set_ylabel("Y Offset")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.axis('equal')
    ax1.legend()

    # Plot 2: Time Series
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(t, x, label='X offset')
    ax2.plot(t, y, label='Y offset', linestyle='--')
    ax2.set_title("Time Series")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Mark the 1s start line
    ax2.axvline(1.0, color='red', linestyle='--', alpha=0.5, label='Start Shifting')

    # Plot 3: Polar Histogram (Radar Plot of Directions)
    ax3 = fig.add_subplot(2, 2, 3, projection='polar')

    if len(unique_angles) > 0:
        bins = np.deg2rad(np.arange(0, 360 + 45, 45))
        hist, _ = np.histogram(np.deg2rad(unique_angles), bins=bins)

        width = np.pi / 4
        ax3.bar(bins[:-1], hist, width=width, bottom=0.0, color='orange', alpha=0.5, edgecolor='black')

    ax3.set_title("Distribution of Directions (Active Periods)")
    ax3.set_theta_zero_location('E')

    plt.tight_layout()
    plt.show()


# --- Detection Function ---

def find_first_direction_indices(pos_seq, target_amplitudes, start_idx=0, tolerance=1e-3, min_separation=5,
                                 debug_flag=False):
    """
    Scans the position sequence to find the first time step for each unique
    (Direction, Amplitude) pair.

    Parameters:
    - pos_seq: (N, 2) array of (x, y) shifts.
    - target_amplitudes: list of floats, expected magnitudes to detect.
    - start_idx: int, frame index to start searching from.
    - tolerance: float, matching tolerance.
    - min_separation: int, minimum frame distance between any accepted indices.
    - debug_flag: bool, visualize results.

    Returns:
    - found_indices: dict, {(angle_int, amplitude_float): index}.
      * angle_int -1 represents "Stay" (0 shift).
      * angle_int 0..315 represents directions.
    """

    # Handle single float input
    if isinstance(target_amplitudes, (int, float)):
        target_amplitudes = [float(target_amplitudes)]

    # Standard 8 directions + Stay (-1)
    directions = [-1, 0, 45, 90, 135, 180, 225, 270, 315]

    found_indices = {}

    debug_points = []  # list of (x, y, angle, amp, idx)
    all_found_indices = []

    for idx in range(start_idx, len(pos_seq)):
        x, y = pos_seq[idx]
        magnitude = np.sqrt(x ** 2 + y ** 2)

        detected_angle = None
        detected_amp = None
        is_match = False

        # Check 1: Is it a "Stay" (Zero Shift)?
        if np.abs(magnitude) < tolerance:
            detected_angle = -1
            detected_amp = 0.0
            is_match = True

        # Check 2: Does it match any target amplitude?
        else:
            for target in target_amplitudes:
                # Use absolute difference to catch peaks for both Square and Sin (near peak)
                if np.abs(magnitude - target) < tolerance:
                    detected_amp = target
                    is_match = True
                    break  # Matched an amplitude

            if is_match:
                # Calculate angle
                rad = np.arctan2(y, x)
                deg = np.degrees(rad)
                if deg < 0: deg += 360
                detected_angle = int(round(deg / 45.0) * 45)
                if detected_angle == 360: detected_angle = 0

        # If we found a candidate match
        if is_match and detected_angle is not None:
            key = (detected_angle, detected_amp)

            # Check if we already found this specific (Dir, Amp) pair
            if key not in found_indices:

                # SEPARATION CHECK:
                too_close = False
                for existing_idx in all_found_indices:
                    if np.abs(idx - existing_idx) < min_separation:
                        too_close = True
                        break

                if not too_close:
                    found_indices[key] = idx
                    all_found_indices.append(idx)
                    debug_points.append((x, y, detected_angle, detected_amp, idx))

    # --- Visualization ---
    if debug_flag:
        plt.figure(figsize=(10, 8))
        plt.plot(pos_seq[:, 0], pos_seq[:, 1], color='lightgray', alpha=0.5, label='Trajectory', zorder=1)
        plt.scatter(pos_seq[:, 0], pos_seq[:, 1], color='gray', s=5, alpha=0.2, zorder=1)
        plt.scatter(0, 0, c='black', marker='x', s=100, label='Origin', zorder=2)

        colors = plt.cm.hsv(np.linspace(0, 1, 10))

        for (dx, dy, dang, damp, didx) in debug_points:
            if dang == -1:
                label_str = f"Stay (idx {didx})"
                c = 'black'
            else:
                c_idx = int(dang // 45)
                label_str = f"{dang}°, A={damp:.1f} (idx {didx})"
                c = colors[c_idx]

            plt.scatter(dx, dy, color=c, s=150, edgecolors='white',
                        label=label_str, zorder=3, marker='o')
            plt.text(dx, dy, str(didx), fontsize=9, ha='center', va='center', color='cyan', fontweight='bold')

        plt.title(f"First Detected Indices (Start: {start_idx})")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    return found_indices


# --- Utility: Count Shift Periods ---

def get_shift_periods_count(trajectory, frequency, sampling_rate=100):
    """
    Calculates the number of active shift periods in a generated trajectory.
    Assumes the trajectory was generated with the 1-second initial delay.

    Parameters:
    - trajectory: np.ndarray, the (N, 2) shift array.
    - frequency: float, cycle frequency used in generation.
    - sampling_rate: int, Hz.

    Returns:
    - count: int, number of active shift phases.
    """
    N = len(trajectory)
    start_active_idx = int(1.0 * sampling_rate)

    if N <= start_active_idx:
        return 0

    dt = 1.0 / sampling_rate
    # Calculate time relative to the start of the active phase (1.0s)
    # The last time point in the array corresponds to (N-1)*dt
    max_time_relative = (N - 1) * dt - 1.0

    if max_time_relative < 0:
        return 0

    # Calculate event index for the last time point
    events_per_second = 2 * frequency
    max_event_idx = int(np.floor(max_time_relative * events_per_second))

    # Active periods correspond to even indices (0, 2, 4...)
    # We want to know how many even integers are <= max_event_idx
    # 0 -> 1 (0)
    # 1 -> 1 (0)
    # 2 -> 2 (0, 2)
    # 3 -> 2 (0, 2)
    # Formula: (index // 2) + 1

    if max_event_idx < 0:
        return 0

    count = (max_event_idx // 2) + 1
    return count


# --- Kernel Alignment Function (Unchanged) ---
def align_kernels_by_shift(kernels_arr, shifts_rcd):
    """
    Pads and shifts a stack of kernels based on (x, y) offsets.
    """
    if not isinstance(kernels_arr, np.ndarray) or kernels_arr.ndim != 3:
        raise ValueError("Input 'kernels_arr' must be a 3D NumPy array [N, size, size].")

    N, original_size, _ = kernels_arr.shape
    if kernels_arr.shape[2] != original_size:
        raise ValueError("Kernels must be square.")

    rounded_shifts = shifts_rcd.astype(int)
    max_shift_x = np.max(np.abs(rounded_shifts[:, 0]))
    max_shift_y = np.max(np.abs(rounded_shifts[:, 1]))
    pad_amount = int(max(max_shift_x, max_shift_y))

    new_size = original_size + 2 * pad_amount
    shifted_kernels_arr = np.zeros((N, new_size, new_size), dtype=kernels_arr.dtype)
    base_row = pad_amount
    base_col = pad_amount

    for i in range(N):
        dx = rounded_shifts[i, 0]
        dy = rounded_shifts[i, 1]
        r_start = base_row + dy
        c_start = base_col + dx
        r_end = r_start + original_size
        c_end = c_start + original_size
        shifted_kernels_arr[i, r_start:r_end, c_start:c_end] = kernels_arr[i]

    return shifted_kernels_arr


def visualize_aligned_kernels(padded_kernels, shifts_rcd, titles=None):
    """Visualization of the padded kernels"""
    N = padded_kernels.shape[0]
    cols = min(N, 4)
    rows = (N + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    fig.suptitle("Aligned Kernels (Padded)", fontsize=16)

    if N == 1: axes = [axes]
    axes = np.array(axes).flatten()

    for i in range(N):
        ax = axes[i]
        im = ax.imshow(padded_kernels[i], cmap='viridis', origin='upper')
        cy, cx = padded_kernels.shape[1] // 2, padded_kernels.shape[2] // 2
        ax.axhline(cy, color='white', alpha=0.3, linestyle='--')
        ax.axvline(cx, color='white', alpha=0.3, linestyle='--')
        shift_str = f"Shift: ({shifts_rcd[i, 0]:.1f}, {shifts_rcd[i, 1]:.1f})"
        title = titles[i] if titles and i < len(titles) else f"Kernel {i}"
        ax.set_title(f"{title}\n{shift_str}", fontsize=9)
        ax.axis('off')

    for i in range(N, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# --- Example Usage ---
if __name__ == "__main__":
    print("1. Generating SIN Sequence with Variable Amplitudes...")
    N_STEPS = 200
    AMPS = [5.0, 10.0]  # Alternating amplitudes
    FREQ = 1.3

    shifts = generate_shift_sequence(
        time_steps_num=N_STEPS,
        amplitude_seq=AMPS,
        frequency=FREQ,
        pattern="square",
        force_second_down=False,  # Use False to test random direction
        debug_flag=True
    )

    print("\n2. Counting Shift Periods...")
    num_periods = get_shift_periods_count(shifts, FREQ)
    print(f"Total Active Shift Periods Found: {num_periods}")

    print("\n3. Detect Indices...")
    # Passing the list of amplitudes to detect
    found_map = find_first_direction_indices(shifts, target_amplitudes=AMPS, start_idx=0, tolerance=0.5,
                                             debug_flag=True)

    print("\n4. Results:")
    # We expect: Stay, (0deg, 5.0), (90deg, 10.0) based on logic
    # First active (0deg) uses AMPS[0] -> 5.0
    # Second active (90deg) uses AMPS[1] -> 10.0

    # Sort keys for clean printing
    sorted_keys = sorted(found_map.keys(), key=lambda k: (k[0], k[1]))

    valid_indices = []

    for (ang, amp) in sorted_keys:
        idx = found_map[(ang, amp)]
        print(f"Angle {ang:>3}°, Amp {amp:>4.1f}: Index {idx}")
        valid_indices.append(idx)

    # Proceed to padding example if indices found
    if valid_indices:
        print("\n5. Padding Example...")
        k_size = 21
        sigma = 3.0
        x = np.linspace(-k_size // 2, k_size // 2, k_size)
        y = np.linspace(-k_size // 2, k_size // 2, k_size)
        xx, yy = np.meshgrid(x, y)
        gaussian = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))

        kernels_stack = np.array([gaussian for _ in valid_indices])
        specific_shifts = shifts[valid_indices]

        padded_stack = align_kernels_by_shift(kernels_stack, specific_shifts)

        titles = [f"{k[0]}°, A={k[1]}" for k in sorted_keys]
        visualize_aligned_kernels(padded_stack, specific_shifts, titles)