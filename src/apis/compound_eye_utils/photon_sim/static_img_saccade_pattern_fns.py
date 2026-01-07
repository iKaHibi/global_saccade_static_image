import numpy as np
import pandas as pd
import random
import os

import matplotlib.pyplot as plt

#TODO: CAUTION: we currently only consider one direction global saccade (saccade offset >= 0)

"""
All scd_ptn_fn (saccade pattern functions) should have same input and output format:
input:
    t: float, in second
output:
    offset_degree: float, in degree, the saccade offset degree 
                    *** (BETTER IN RANGE [0, 3.2] for reasonable skewed kernel) ***
"""

# Get the directory of the current Python file
current_file_dir = os.path.dirname(os.path.abspath(__file__))

# we used Class in case of experiment saccade pattern record to avoid opening .xlsx files for multiple times
class SaccadeRcdContainer:
    def __init__(self, saccade_rcd_type: str):
        if saccade_rcd_type == "static_grating":
            """
            the saccade pattern function for static low frequency grating images
                corresponds to Nat22 Fig4.b
            this will randomly select one saccade record of right eye from Nat22 data
            """
            self.type = "static_grating"
            file_path = os.path.join(current_file_dir, "src_files",
                                     "exp_saccade_patterns", "Figure4.xlsx")
            target_sheet_name = 'saccades_vertical_grating'
            # Read the data from the .xlsx file into a pandas DataFrame
            self.data = pd.read_excel(file_path, sheet_name=target_sheet_name)
            random_idx = random.randint(0, 120)
            self.scd_pattern_rcd = np.abs(self.data.values[3:, random_idx])
            self.max_offset_degree = np.max(self.scd_pattern_rcd)
            self.rcd_len = len(self.scd_pattern_rcd)
            self.rcd_fps = 1000
        elif saccade_rcd_type == "static_grating_idx":
            self.type = "static_grating"
            file_path = os.path.join(current_file_dir, "src_files",
                                     "exp_saccade_patterns", "Figure4.xlsx")
            target_sheet_name = 'saccades_vertical_grating'
            # Read the data from the .xlsx file into a pandas DataFrame
            self.data = pd.read_excel(file_path, sheet_name=target_sheet_name)
            self.scd_pattern_rcd = np.abs(self.data.values[3:, 0])
            self.max_offset_degree = np.max(self.scd_pattern_rcd)
            self.rcd_len = len(self.scd_pattern_rcd)
            self.rcd_fps = 1000
        elif saccade_rcd_type == "static_darkness":
            self.type = "static_darkness"
            file_path = os.path.join(current_file_dir, "src_files",
                                     "exp_saccade_patterns", "static_dark_rcd.npy")
            self.data = np.load(file_path, allow_pickle=True)
            random_idx = random.randint(0, len(self.data)-1)
            self.scd_pattern_rcd = np.abs(self.data[random_idx])
            self.max_offset_degree = np.max(self.scd_pattern_rcd)
            self.rcd_len = len(self.scd_pattern_rcd)
            self.rcd_fps = 1000
        elif saccade_rcd_type == "moving_grating":
            self.type = "moving_grating"
            file_path = os.path.join(current_file_dir, "src_files",
                                     "exp_saccade_patterns", "Figure3.xlsx")
            target_sheet_name = 'full_righward'
            self.data = pd.read_excel(file_path, sheet_name=target_sheet_name)
            random_idx = random.randint(0, 10)
            self.scd_pattern_rcd = np.abs(self.data.values[3:, random_idx])
            desired_length = 10000
            # Check if the array needs to be extended
            if len(self.scd_pattern_rcd) < desired_length:
                # Calculate the number of values to add
                num_values_to_add = desired_length - len(self.scd_pattern_rcd)
                # Create an array of the final value repeated `num_values_to_add` times
                repeated_values = np.full(num_values_to_add, self.scd_pattern_rcd[-1])
                # Concatenate the original array with the repeated values
                self.scd_pattern_rcd = np.concatenate((self.scd_pattern_rcd, repeated_values))
            self.max_offset_degree = np.max(self.scd_pattern_rcd)
            self.rcd_len = len(self.scd_pattern_rcd)
            self.rcd_fps = 1000
        else:
            raise BaseException("Unknown saccade_rcd type")

    def reset_scd_pattern_rcd(self, chosen_rcd_idx=-1):
        if self.type == "static_grating":
            random_idx = random.randint(0, 120)
            self.scd_pattern_rcd = np.abs(self.data.values[3:, random_idx])
            self.max_offset_degree = np.max(self.scd_pattern_rcd)
            self.rcd_len = len(self.scd_pattern_rcd)
        elif self.type == "static_grating_idx":
            assert chosen_rcd_idx < 147, "Chosen Record Index Out of Range (>147)"
            self.scd_pattern_rcd = np.abs(self.data.values[3:, chosen_rcd_idx])
            self.max_offset_degree = np.max(self.scd_pattern_rcd)
            self.rcd_len = len(self.scd_pattern_rcd)
        elif self.type == "static_darkness":
            random_idx = random.randint(0, len(self.data) - 1)
            self.scd_pattern_rcd = np.abs(self.data[random_idx])
            self.max_offset_degree = np.max(self.scd_pattern_rcd)
            self.rcd_len = len(self.scd_pattern_rcd)

    def scd_ptn_fn(self, t: float):
        rcd_idx = int(t * self.rcd_fps % self.rcd_len)
        return self.scd_pattern_rcd[rcd_idx]

class SaccadePatternGenerator:
    def __init__(self, saccade_rcd_type: str, saccade_amp=-1):
        if saccade_rcd_type == "static_darkness":
            """
            the saccade pattern function for static darkness image fitted curve
                corresponds to Nat22 Fig4.a
            this is based on fitted curve function in fig4_data_extr.py
            """
            rcd_length = 1999
            time_step = 1e-3  # in second
            time_seq = np.arange(0, rcd_length * time_step, time_step)
            saccade_start_idx = 950
            saccade_end_idx = 1040
            saccade_start_idx = 950
            saccade_end_idx = 1040
            ts_away = np.array(time_seq[saccade_start_idx:saccade_end_idx]) - time_seq[saccade_start_idx]
            fitted_away_funiton = lambda t: -20.814734786792002 * t
            # artificial backward shifting function (always positive)
            ts_back = np.array(time_seq[saccade_end_idx:]) - time_seq[saccade_end_idx]
            fitted_back_function = lambda t: 1.8542803022168592 * np.exp(-6 * t)
            # getting the whole fitted curve
            fitted_curve = np.zeros(rcd_length)
            fitted_curve[:saccade_start_idx] = 0
            fitted_curve[saccade_start_idx:saccade_end_idx] = fitted_away_funiton(ts_away)
            fitted_curve[saccade_end_idx:] = -1 * fitted_back_function(ts_back)
            self.scd_pattern_rcd = np.abs(fitted_curve) / np.max(np.abs(fitted_curve)) * 3.667  # TODO: check this
            self.max_offset_degree = np.max(self.scd_pattern_rcd)
            self.rcd_len = len(self.scd_pattern_rcd)
            self.rcd_fps = 1000
        elif saccade_rcd_type == "static":
            # rcd_length = 1999
            # idx_seq = np.linspace(0, rcd_length, rcd_length)
            # # getting the whole fitted curve
            # fitted_curve = idx_seq % 10 / 10

            rcd_length = 1000 # 1999
            idx_seq = np.linspace(0, rcd_length, rcd_length)
            # getting the whole fitted curve
            fitted_curve = np.zeros_like(idx_seq, dtype=np.float32)
            fitted_curve[:500] = 1.

            self.max_offset_degree = 1.7  #2.88
            self.scd_pattern_rcd = np.abs(fitted_curve) / np.max(np.abs(fitted_curve)) * self.max_offset_degree #1.7  # TODO: check this
            self.max_offset_degree = np.max(self.scd_pattern_rcd)
            self.rcd_len = len(self.scd_pattern_rcd)
            self.rcd_fps = 1000
        elif saccade_rcd_type == "moving_grating":
            """
            the saccade pattern function for moving grating fitted curve
                corresponds to Nat22 Fig3.a
            this is based on fitted curve function in fig3_data_extr.py
            *** CAUTION: THIS REQUIRES THE SIMULATION LENGTH TO >4s and <10s *** 
            """
            rcd_length = 11000
            time_step = 1e-3  # in second
            time_seq = np.arange(0, rcd_length * time_step, time_step)
            saccade_start_idx = 1500
            saccade_end_idx = 3400
            ts = np.array(time_seq[saccade_start_idx:saccade_end_idx]) - time_seq[saccade_start_idx]
            fitted_funiton = lambda t: -0.44030537 * t ** 2 + 2.59324275 * t
            # getting the whole fitted curve
            fitted_curve = np.zeros(rcd_length)
            fitted_curve[:saccade_start_idx] = 0
            fitted_curve[saccade_start_idx:saccade_end_idx] = fitted_funiton(ts)
            fitted_curve[saccade_end_idx:] = fitted_curve[saccade_end_idx - 1]
            if saccade_amp < 0:
                self.scd_pattern_rcd = np.abs(fitted_curve)
            else:
                self.scd_pattern_rcd = np.abs(fitted_curve) / np.max(np.abs(fitted_curve)) * saccade_amp
            self.max_offset_degree = np.max(self.scd_pattern_rcd)
            self.rcd_len = len(self.scd_pattern_rcd)
            self.rcd_fps = 1000
        else:
            raise BaseException("Unknown saccade pattern type")

    def scd_ptn_fn(self, t: float):
        rcd_idx = int(t * self.rcd_fps % self.rcd_len)
        return self.scd_pattern_rcd[rcd_idx]

if __name__ == "__main__":
    # test_target = "rcd_moving_grating"
    # test_target = "static_darkness_fit"
    test_target = "moving_grating"
    fps = 100
    ts = np.linspace(0, 10, 10 * fps)
    if test_target == "static_grating":
        # testing Static Low Frequency Grating
        for _ in range(3):
            static_low_freq_grating_scd_container = SaccadeRcdContainer(saccade_rcd_type="static_grating")
            scd_rcd = np.zeros_like(ts)
            print("initialization done")
            for idx, t in enumerate(ts):
                scd_rcd[idx] = static_low_freq_grating_scd_container.scd_ptn_fn(t)

            plt.plot(ts[::10], scd_rcd[::10], alpha=0.7)
        plt.xticks(fontsize=14)
        plt.xlabel("time", fontsize=16)
        plt.ylabel("Degree", fontsize=16)
        plt.title("Random Sampled Experiment Saccade Record", fontsize=16)
        plt.show()
    elif test_target == "static_darkness":
        for _ in range(4):
            static_darkness_scd_container = SaccadeRcdContainer(saccade_rcd_type="static_darkness")
            scd_rcd = np.zeros_like(ts)
            print("initialization done")
            for idx, t in enumerate(ts):
                scd_rcd[idx] = static_darkness_scd_container.scd_ptn_fn(t)

            plt.plot(ts[::10], scd_rcd[::10]*static_darkness_scd_container.max_offset_degree, alpha=0.7)
        plt.xlabel("time")
        plt.ylabel("saccade record offset")
        plt.show()
    elif test_target == "static_darkness_fit":
        static_darkness_scd_generator = SaccadePatternGenerator(saccade_rcd_type="static_darkness")
        scd_rcd = np.zeros_like(ts)
        print("initialization done")
        for idx, t in enumerate(ts):
            scd_rcd[idx] = static_darkness_scd_generator.scd_ptn_fn(t)*static_darkness_scd_generator.max_offset_degree

        plt.plot(ts, scd_rcd)
        plt.xlabel("time")
        plt.ylabel("saccade record offset")
        plt.show()
    elif test_target == "moving_grating":
        moving_grating_scd_generator = SaccadePatternGenerator(saccade_rcd_type="moving_grating")
        scd_rcd = np.zeros_like(ts)
        print("initialization done")
        for idx, t in enumerate(ts):
            scd_rcd[idx] = moving_grating_scd_generator.scd_ptn_fn(t)
        print(max(scd_rcd))

        plt.plot(ts, scd_rcd)
        plt.xlabel("time")
        plt.ylabel("saccade record offset")
        plt.show()
    elif test_target == "rcd_moving_grating":
        for _ in range(4):
            moving_grating_scd_generator = SaccadeRcdContainer(saccade_rcd_type="moving_grating")
            scd_rcd = np.zeros_like(ts)
            print("initialization done")
            for idx, t in enumerate(ts):
                scd_rcd[idx] = moving_grating_scd_generator.scd_ptn_fn(t)

            plt.plot(ts[:795:10], scd_rcd[:795:10], alpha=0.7)
        plt.xlabel("time")
        plt.ylabel("saccade record offset")
        plt.show()
