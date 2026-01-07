import os
import ctypes
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from .static_img_saccade_pattern_fns import *
from .photon_utils import *

from ..random_dir_shift_util import generate_shift_sequence, get_shift_periods_count

# Get the directory of the current Python file
current_file_dir = os.path.dirname(os.path.abspath(__file__))

SHIFT_AMOUNT_PIX = 1

def plot_and_save_positions(pos_arr, color_num):
    """
    Plots a set of (x, y) positions using a scatter plot and saves the figure.

    The first `color_num` points in the pos_arr are plotted in red, and the
    remaining points are plotted in blue.

    Args:
        pos_arr (np.ndarray): A numpy array of shape (N, 2), where N is the
                              total number of points. Each row represents an
                              (x, y) coordinate.
        color_num (int): The number of initial points to color red.
    """
    # --- Input Validation ---
    if not isinstance(pos_arr, np.ndarray) or pos_arr.ndim != 2 or pos_arr.shape[1] != 2:
        raise ValueError("pos_arr must be a 2D numpy array with shape (N, 2).")
    if not isinstance(color_num, int) or color_num < 0:
        raise ValueError("color_num must be a non-negative integer.")

    # Ensure the output directory exists
    output_dir = "tests"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, "test_pos.png")

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(8, 8))

    # Split the data into red and blue points
    red_points = pos_arr[:color_num]
    blue_points = pos_arr[color_num:]

    # Plot the red points, if any
    if red_points.shape[0] > 0:
        ax.scatter(red_points[:, 0], red_points[:, 1], c='red', label=f'First {color_num} Points', s=10)

    # Plot the blue points, if any
    if blue_points.shape[0] > 0:
        ax.scatter(blue_points[:, 0], blue_points[:, 1], c='blue', label='Remaining Points', s=10)

    # --- Formatting and Saving ---
    ax.set_title("Scatter Plot of Positions")
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_aspect('equal', adjustable='box') # Ensure aspect ratio is equal

    # Save the figure to the specified file
    plt.savefig(output_path)
    plt.close(fig) # Close the figure to free up memory
    print(f"Plot successfully saved to: {output_path}")



class PhotonFilter:
    def __init__(self, saccade_pattern_generator, offset_kernel_gen_fn, omm_num_sqrt=32, fps=100, saving_path_dict=None):
        """
        saccade_pattern_generator: class with method `scd_ptn_fn`
            `max_offset_degree` member:
                the max offset degree in float, for kernel generation
            `scd_ptn_fn` method:
                function takes in time t (in second), returning saccade degree (float)
        offset_kernel_gen_fn: function
                input:
                    hw(float, controlling the kernel size)
                    saccade degree (float, in degree),
                output:
                    2d numpy array, the skewed_kernel_generator
        """
        self.saccade_pattern_generator = saccade_pattern_generator
        self.offset_kernel_gen_fn = offset_kernel_gen_fn
        self.fps = fps

        # generate variables for ommatidia kernel
        self.omm_num = omm_num_sqrt ** 2
        # get ommatidia positions
        pos_arr, _ = get_square_pos(omm_num_sqrt, mode="Grid")
        self.pos_arr_without_scaling = pos_arr

        self._display_frame_num_threshold = 100

        if saving_path_dict is None:
            self._src_video_saving_path = None
            self._kernel_saving_path = None
        else:
            if "src_video_saving_path" in saving_path_dict.keys():
                self._src_video_saving_path = saving_path_dict["src_video_saving_path"]
            if "kernel_saving_path" in saving_path_dict.keys():
                self._kernel_saving_path = saving_path_dict["kernel_saving_path"]


    @staticmethod
    def get_kernel_size(src_img_size, omm_num_sqrt, small_ns_kernel=False):
        pos_arr, _ = get_square_pos(omm_num_sqrt, mode="full")
        pos_arr = pos_arr * src_img_size // 2
        pos_arr = np.round(pos_arr)

        omm_dist = np.abs(pos_arr[0, 0] - pos_arr[1, 0])
        hw_size = int((omm_dist / 5. * 8.9) / 2)
        kernel_size = int(hw_size * 1.2) * 2 + 1

        return kernel_size

    def filter_based_on_omm(self, src_arr, src_frame_fn, img_size=2000, time_len=1.,
                            with_saccade=False, with_jitter=False, shift_pixel_list=[0], shift_degree_list=[0.], shift_freq=None, shift_pattern="square"):
        # set parameters
        total_frames = int(time_len * self.fps)
        # img_scaling = img_size
        img_scaling = img_size * 0.9

        # preparing ommatidia positions array
        pos_arr = self.pos_arr_without_scaling * img_scaling // 2
        pos_arr[:, 0] += src_frame_fn(src_arr, 0).shape[0] // 2
        pos_arr[:, 1] += src_frame_fn(src_arr, 0).shape[1] // 2
        pos_arr = np.round(pos_arr)

        # the frame array size
        src_frame_shape = src_frame_fn(src_arr, 0).shape
        assert src_frame_shape[0] == src_frame_shape[1], \
            "The Source Frame (after preprocess and changing function) should have width == height"
        assert src_frame_shape[0] == img_size, \
            f"The `img_size`({img_size}) should be same as Source Frame size({src_frame_shape[0]})"

        # generate different filtering kernels
        omm_dist = np.abs(pos_arr[0, 0] - pos_arr[1, 0])

        photon_res = np.zeros([total_frames, self.omm_num])

        # record the saccade pattern / source rotate image for debug
        offset_dgr_rcd = np.zeros(total_frames)
        src_frame_rcd = np.zeros([total_frames, src_frame_shape[0], src_frame_shape[1]])
        pos_used_rcd = np.zeros([total_frames, pos_arr.shape[0], pos_arr.shape[1]])
        shift_pix_float_rcd = np.zeros([total_frames, 2])

        # generate shift trajectory
        jitter_pattern_seq = generate_shift_sequence(total_frames, shift_pixel_list, shift_freq, pattern=shift_pattern)
        num_shift_periods = get_shift_periods_count(jitter_pattern_seq, shift_freq)

        # 找到当前 trajectory 下，对应的最大 shift pixel 和最大 shift degree
        scale_period_idx = np.min((num_shift_periods, len(shift_degree_list)))
        max_shift_degree = np.max(shift_degree_list[:scale_period_idx])
        max_shift_pixel = np.max(shift_pixel_list[:scale_period_idx])
        pixel2degree_scaling = max_shift_degree / max_shift_pixel if max_shift_pixel != 0 else 0.

        # generate different filtering kernels
        omm_dist = np.abs(pos_arr[0, 0] - pos_arr[1, 0])
        hw_size = int((omm_dist / 5.5 * 8.) / 2)  # CAUTION: small FWHM test, other wise * 8.

        kernels_arr = self._generate_diff_kernels(hw_size, shift_degree=max_shift_degree)

        kernel_arr_used_rcd = np.zeros((total_frames, kernels_arr.shape[-2], kernels_arr.shape[-1]))

        # iterate to process all frames
        total_frames_iterator = range(total_frames)
        for frame_idx in total_frames_iterator:
            # getting the offset scale for each ommatidia
            if with_saccade and not with_jitter:  # never reached
                raise BaseException("No Shift, Only Shrink not implemented.")
            elif with_jitter and not with_saccade:
                # only the jitter offset
                kernel_idx_to_use_arr = np.array([0.] * self.omm_num)
                jitter_offset_pixel_float = jitter_pattern_seq[frame_idx]
            elif with_saccade and with_jitter:
                # both saccade and jitter offset
                jitter_offset_pixel_float = jitter_pattern_seq[frame_idx]
                saccade_offset_degree = (np.sqrt(jitter_offset_pixel_float[0] ** 2 + jitter_offset_pixel_float[1] ** 2)
                                         * pixel2degree_scaling)  # pixel 转为 degree，当前的shift degree
                offset_dgr_rcd[frame_idx] = saccade_offset_degree
                kernel_idx_to_use = round(saccade_offset_degree / max_shift_degree * 10) / 10  # 由于 skewed kernel array 是根据 max_shift 决定的，所以用这个
                kernel_idx_to_use_arr = np.array([kernel_idx_to_use] * self.omm_num)
            else:
                kernel_idx_to_use_arr = np.array([0.] * self.omm_num)
                jitter_offset_pixel_float = np.array([0., 0.])

            # saving the kernel idx used if needed
            kernel_arr_used_rcd[frame_idx] = kernels_arr[np.min([int(kernel_idx_to_use_arr[0] * 10), 9])]

            # CAUTION: 可移植性更好的做法是设置 jitter_offset_degree，然后转成pixel，但由于我们这个测试的输入图片永远是 1080pixel，所以为了后续reconstruciton方便，定位 jitter max offset = 2 pixel
            pos_arr_used = pos_arr + np.array(jitter_offset_pixel_float, dtype=int)
            pos_used_rcd[frame_idx] = pos_arr_used
            shift_pix_float_rcd[frame_idx] = jitter_offset_pixel_float

            # do the photon filter
            frame2process = src_frame_fn(src_arr, frame_idx)
            photon_res[frame_idx, :] = PhotonFilter.img2photon_all_omm(frame2process, pos_arr_used,
                                                                       self.omm_num,
                                                                       kernel_idx_to_use_arr, kernels_arr)

            src_frame_rcd[frame_idx] = np.mean(frame2process, axis=2)

        return photon_res, pos_used_rcd, shift_pix_float_rcd, src_frame_rcd, kernel_arr_used_rcd

    @staticmethod
    def filter_one_frame(frame_arr, omm_num_sqrt, normed_offset_arr=None, kernels_arr=None, offset_arr=None):
        omm_num = omm_num_sqrt ** 2
        # get ommatidia positions
        pos_arr, _ = get_square_pos(omm_num_sqrt, mode="full")
        pos_arr_without_scaling = pos_arr

        img_scaling = frame_arr.shape[0]
        img_scaling = frame_arr.shape[0] * 0.9

        # preparing ommatidia positions array
        pos_arr = pos_arr_without_scaling * img_scaling // 2
        pos_arr[:, 0] += frame_arr.shape[0] // 2
        pos_arr[:, 1] += frame_arr.shape[1] // 2
        pos_arr = np.round(pos_arr)

        if kernels_arr is None:
            omm_dist = np.abs(pos_arr[0, 0] - pos_arr[1, 0])
            hw_size = int((omm_dist / 5. * 8.9) / 2)
            kernels_arr = PhotonFilter._generate_diff_gaussian_kernels(hw_size)

        if normed_offset_arr is None:
            normed_offset_arr = np.array([0.] * pos_arr.shape[0])

        if offset_arr is not None:
            pos_arr += offset_arr
            # plt.scatter(pos_arr)

        # print("pos_val", pos_arr_without_scaling[:omm_num_sqrt, 1])
        # print("pos_val", pos_arr_without_scaling[omm_num_sqrt:omm_num_sqrt*2, 1])


        # photon_res = PhotonFilter.img2photon_all_omm(frame_arr, pos_arr, omm_num, normed_offset_arr, kernels_arr)

        return PhotonFilter.img2photon_all_omm(frame_arr, pos_arr, omm_num, normed_offset_arr, kernels_arr)

    def filter_img(self, img_arr, img_size=2000, time_len=1,
                   with_saccade=False,
                   img_rotate_speed=None,
                   pre_process_fn=None,
                   with_jitter=False,
                   shift_pixel_list=[0],
                   shift_degree_list=[0.],
                   shift_freq=None,
                   shift_pattern="square"
                   ):
        """
        img_rotate_speed: float, the speed of image rotation in °/s
        """
        # reading source image
        src_img = img_convert_to_3channels(img_arr, img_size)
        if pre_process_fn is not None:
            src_img = pre_process_fn(src_img)
        src_img_take_frame_fn = lambda src_arr, frame_idx: src_arr
        if img_rotate_speed is not None:
            img_rotate_fn = lambda src_arr, frame_idx: \
                image_left_to_right_rotate_dgr(src_arr, frame_idx / self.fps * img_rotate_speed)
            combined_img_fn = lambda src_arr, frame_idx: \
                img_rotate_fn(src_img_take_frame_fn(src_arr, frame_idx), frame_idx)
        else:
            combined_img_fn = src_img_take_frame_fn

        photon_res, pos_used_rcd, shift_pix_float_rcd, rotate_src_img_seq, kernel_used = self.filter_based_on_omm(src_img, combined_img_fn,
                                                              img_size, time_len, with_saccade, with_jitter=with_jitter,
                                                                                             shift_pixel_list=shift_pixel_list,
                                                                                             shift_degree_list=shift_degree_list,
                                                                                             shift_freq=shift_freq,
                                                                                             shift_pattern=shift_pattern)

        return photon_res, pos_used_rcd, shift_pix_float_rcd, rotate_src_img_seq, kernel_used

    @staticmethod
    def _generate_diff_gaussian_kernels(hw_size):
        """
        generating `kernel_num` different kernels
            kernel_idx==0: no offset gaussian kernel
        """
        kernel_num = 10  # it cannot be changed currently

        kernel_list = []
        for _ in range(kernel_num):
            kernel_list.append(generate_gaussian_kernel(hw_size))
        return np.array(kernel_list)

    def _generate_diff_kernels(self, hw_size, shift_degree):
        """
        generating `kernel_num` different kernels
            kernel_idx==0: no offset gaussian kernel
            kernel_idx==[1, 9]: skewed offset gaussian kernel, with maximum offset degree
        """
        kernel_num = 10  # it cannot be changed currently

        kernel_list = []
        for idx in range(kernel_num):
            if idx == 0:
                kernel_list.append(generate_gaussian_kernel(hw_size))
            else:
                offset_degree = idx/(kernel_num-1) * shift_degree
                kernel = self.offset_kernel_gen_fn(hw_size, offset_degree=offset_degree)
                kernel_list.append(kernel)
        return np.array(kernel_list)

    def _generate_diff_kernels_only_offset(self, hw_size):
        """
        generating `kernel_num` different kernels, only have receptive field offset, no kernel shrink
            kernel_idx==[0, 9]: no offset gaussian kernel
        """
        kernel_num = 10  # it cannot be changed currently

        kernel_list = []
        for idx in range(kernel_num):
            if idx == 0:
                kernel_list.append(generate_gaussian_kernel(hw_size, ns_small_kernel=self._small_ns_kernel))
            else:
                # TODO: recheck this process
                offset_degree = idx / (kernel_num - 1) * self.saccade_pattern_generator.max_offset_degree
                kernel = generate_shifted_gaussian_kernel(hw_size, offset_degree=offset_degree)
                kernel_list.append(kernel)
        return np.array(kernel_list)

    @staticmethod
    def img2photon_all_omm(img, pos_arr, omm_num, normed_offset_arr, kernels_arr):
        kernel_num = kernels_arr.shape[0]
        kernel_size = kernels_arr.shape[1]

        img_input_resize = img.flatten()
        kernels_input_resize = kernels_arr.flatten()

        img_size = img.shape[0]
        if img_size ** 2 != img_input_resize.shape[0] // 3:
            raise BaseException(f"The Image Is Not Square! (width, height)=({img.shape[0]}, {img.shape[1]})")

        img_input_resize = np.array(img_input_resize, dtype=np.float32)
        kernels_input_resize = np.array(kernels_input_resize, dtype=np.float32)
        photon_output = np.zeros(omm_num, dtype=np.float32)

        pos_arr_resize = np.array(np.round(pos_arr.flatten()), dtype=np.int32)
        normed_offset_arr_resized = np.array(normed_offset_arr.flatten(), dtype=np.float32)

        # Load the shared object file
        # Extract the "../c_lib" path relative to current_file_dir
        parent_dir = os.path.dirname(current_file_dir)
        shared_lib_path = os.path.join(parent_dir, "libs", "img2photon_square_omm.so")
        lib = ctypes.CDLL(shared_lib_path)  # Replace with the path to your .so file
        # Define the argument and return types for the function
        lib.filter_one_img.argtypes = [ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_int),
                                       ctypes.c_int,
                                       ctypes.POINTER(ctypes.c_float),
                                       ctypes.c_int,
                                       ctypes.POINTER(ctypes.c_float),
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.POINTER(ctypes.c_float)
                                       ]

        lib.filter_one_img.restype = None
        # Convert the Python lists or numpy arrays to C-compatible pointers
        img_input_ptr = np.ctypeslib.as_ctypes(img_input_resize)
        kernels_input_ptr = np.ctypeslib.as_ctypes(kernels_input_resize)
        pos_arr_input_ptr = np.ctypeslib.as_ctypes(pos_arr_resize)
        normed_offset_input_ptr = np.ctypeslib.as_ctypes(normed_offset_arr_resized)
        photon_output_ptr = np.ctypeslib.as_ctypes(photon_output)

        # Call the CUDA function
        # print("calling C extern funciton for photon filtering")
        lib.filter_one_img(img_input_ptr,
                           pos_arr_input_ptr,
                           omm_num,
                           normed_offset_input_ptr,
                           img_size,
                           kernels_input_ptr,
                           kernel_size,
                           kernel_num,
                           photon_output_ptr)

        # photon_input = np.ctypeslib.as_array(photon_input_ptr, shape=len(photon_input))
        photon_output = np.ctypeslib.as_array(photon_output_ptr, shape=len(photon_output))

        return photon_output

if __name__ == '__main__':
    # saccade_pattern_generator = SaccadeRcdContainer("static_grating")
    # offset_kernel_gen_fn = generate_skewed_gaussian_kernel_fitting_exp
    #
    # photon_filter = PhotonFilter(saccade_pattern_generator, offset_kernel_gen_fn)
    # print(PhotonFilter.get_kernel_size(128, 32))
    from PIL import Image

    omm_sqrt = 320
    offset_arr = np.array([0, 2])
    img_path = "data/mcgill_preprocessed/Flowers_0107.png"
    image_array = np.array(Image.open(img_path))
    image_grey = img_convert_to_3channels(image_array, 720)
    plt.imsave("tests/test_src.png", image_grey, cmap="gray")
    sampled_value = PhotonFilter.filter_one_frame(image_grey, omm_sqrt, offset_arr=offset_arr)
    print(sampled_value.shape)
    sampled_img = np.resize(sampled_value, (omm_sqrt, omm_sqrt))
    plt.imsave("tests/test_sample.png", sampled_img, cmap="gray")
    