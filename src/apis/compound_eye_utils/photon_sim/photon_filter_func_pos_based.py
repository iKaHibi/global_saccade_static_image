import numpy as np
import re
import os
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.special import erf

from src.apis.compound_eye_utils.photon_sim.photon_filter_pos_based import PhotonFilter
from src.apis.compound_eye_utils.photon_sim.static_img_saccade_pattern_fns import *
# from .photon_sim.photon_filter_pos_based import PhotonFilter
# from .photon_sim.static_img_saccade_pattern_fns import *

def generate_gaussian_kernel(hw_size, debug_flag=False):
    '''
    generate a 2d gaussian kernel given with half-width-size
    :param hw_size: the half-width size of gaussian kernel in degree
    '''

    kernel_size = int(hw_size * 1.2) * 2 + 1  # a bit wider than 2 * hw_size, for latter offset convolution

    x = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
    y = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
    xx, yy = np.meshgrid(x, y)

    sigma = hw_size / (2 * np.sqrt(2 * np.log(2)))

    # Calculate the Gaussian kernel
    gaussian_kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    gaussian_kernel /= (2 * np.pi * sigma ** 2)  # Normalize the kernel

    return gaussian_kernel

def generate_skewed_gaussian_kernel_fitting_exp(hw_size, offset_degree=2.6, debug_flag=False, more_offset=1.55):
    '''
    *** Based On Version in utils_skew_tmp_test.py ***
    generate skewed gaussian according to
    :param hw_size: the half-width size of gaussian kernel in degree
    :param offset_degree: should be in degree
    :param debug_flag:  to display or not
    :return: a 2d array
    '''
    if offset_degree / 2.6 < 1. / 16.:
        return generate_gaussian_kernel(hw_size, debug_flag)

    kernel_size = int(hw_size * 1.2) * 2 + 1  # a bit wider than 2 * hw_size, for latter offset convolution

    # Calculate the standard deviation from the FWHM
    offset_scaling = offset_degree / 2.6
    sigma = hw_size / (2 * np.sqrt(2 * np.log(2))) * (1 - 0.35 * offset_scaling) # / 1.4
    offset = sigma * 0.8 * offset_scaling  # 0.8: shrink, maybe caused by contraction?
    skew = -5 * offset_scaling

    x = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
    y = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
    xx, yy = np.meshgrid(x, y)

    exponent = -0.5 * (((xx - more_offset * offset) / sigma) ** 2 + (yy / sigma) ** 2)
    gaussian_part = np.exp(exponent)
    erf_part = 1 + erf(skew * ((xx - more_offset * offset) / (np.sqrt(2) * sigma)))

    kernel = gaussian_part * erf_part
    skew_kernel = kernel / np.sum(kernel)

    if debug_flag:
        print(np.sum(skew_kernel))
        # plt.imshow(gaussian_part)
        # plt.show()
        plt.xticks([])
        plt.yticks([])
        plt.imshow(skew_kernel)
        plt.show()

    return skew_kernel


def filtering_moving_img(img_array, img_size=1000, sim_len=6, scd_type="ns", with_saccade=False,
                         mov_speed=15, omm_num_sqrt=32):
    assert 4 < sim_len <= 8, "Simulation Length should between 4s and 8s"

    if scd_type == "fit":
        global_saccade_pattern_generator = SaccadePatternGenerator("moving_grating")
    elif scd_type == "record":
        global_saccade_pattern_generator = SaccadeRcdContainer("moving_grating")
        print(f"Using Moving Grating Record, max_offset={global_saccade_pattern_generator.max_offset_degree}")
    elif scd_type == "ns":
        global_saccade_pattern_generator = SaccadePatternGenerator("moving_grating")
    else:
        raise BaseException("scd_type should be 'fit', 'ns' or 'record'")

    offset_kernel_gen_fn = generate_skewed_gaussian_kernel_fitting_exp

    photon_filter = PhotonFilter(global_saccade_pattern_generator, offset_kernel_gen_fn,
                                 omm_num_sqrt=omm_num_sqrt)
    photon_res, saccade_rcd, rotate_src_img_seq, kernel_used = photon_filter.filter_img(img_array,
                                                                                        img_size=img_size,
                                                                                        time_len=sim_len,
                                                                                        img_rotate_speed=mov_speed,
                                                                                        with_saccade=with_saccade)

    return photon_res, saccade_rcd, rotate_src_img_seq, kernel_used


def filtering_static_img(img_array, img_size=1000, sim_len=2, with_saccade=False,
                        omm_num_sqrt=32, with_jitter=None, shift_pixel_list=[0], shift_degree_list=[1.7], shift_freq=None, shift_pattern="square"):
    
    global_saccade_pattern_generator = SaccadePatternGenerator("static")

    offset_kernel_gen_fn = generate_skewed_gaussian_kernel_fitting_exp

    photon_filter = PhotonFilter(global_saccade_pattern_generator, offset_kernel_gen_fn,
                                 omm_num_sqrt=omm_num_sqrt)
    photon_res, pos_rcd, shift_pix_float_rcd, rotate_src_img_seq, kernel_used = photon_filter.filter_img(img_array,
                                                                                    img_size=img_size,
                                                                                    time_len=sim_len,
                                                                                    img_rotate_speed=None,
                                                                                    with_saccade=with_saccade,
                                                                                    with_jitter=with_jitter,
                                                                                    shift_pixel_list=shift_pixel_list,
                                                                                    shift_degree_list=shift_degree_list,
                                                                                    shift_freq=shift_freq,
                                                                                    shift_pattern=shift_pattern)

    return photon_res, pos_rcd, shift_pix_float_rcd, rotate_src_img_seq, kernel_used

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    get_center_line = lambda arr: arr[arr.shape[0]//2, :] / np.max(arr[arr.shape[0]//2, :])

    hw_size = 30
    offset_degrees = [1., 1.6]
    base_kernel = generate_gaussian_kernel(hw_size)
    kernels = []

    plt.plot(get_center_line(base_kernel))
    for idx, offset_degree in enumerate(offset_degrees):
        kernels.append(generate_skewed_gaussian_kernel_fitting_exp(hw_size, offset_degree=offset_degree))

        plt.plot(get_center_line(kernels[idx]))

    plt.show()

    # 保存基础核
    plt.imshow(base_kernel)
    plt.axis('off')  # 关闭坐标轴
    plt.gca().set_axis_off()  # 关闭坐标轴
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)  # 移除边距
    plt.margins(0, 0)  # 移除边距
    plt.gca().xaxis.set_major_locator(plt.NullLocator())  # 移除刻度
    plt.gca().yaxis.set_major_locator(plt.NullLocator())  # 移除刻度
    plt.savefig("base_kernel.png", dpi=100, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()

    # 保存变形后的核
    plt.imshow(kernels[0])
    plt.axis('off')  # 关闭坐标轴
    plt.gca().set_axis_off()  # 关闭坐标轴
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)  # 移除边距
    plt.margins(0, 0)  # 移除边距
    plt.gca().xaxis.set_major_locator(plt.NullLocator())  # 移除刻度
    plt.gca().yaxis.set_major_locator(plt.NullLocator())  # 移除刻度
    plt.savefig("skewed_kernel.png", dpi=100, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()
