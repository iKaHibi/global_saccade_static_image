import numpy as np
from scipy.special import erf

def generate_gaussian_kernel(hw_size):
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


def generate_skewed_gaussian_kernel_fitting_exp(hw_size, offset_degree=2.6, more_offset=1.55):
    '''
    *** Based On Version in utils_skew_tmp_test.py ***
    generate skewed gaussian according to
    :param hw_size: the half-width size of gaussian kernel in degree
    :param offset_degree: should be in degree
    :param debug_flag:  to display or not
    :return: a 2d array
    '''
    if offset_degree / 2.6 < 1. / 16.:
        return generate_gaussian_kernel(hw_size)

    kernel_size = int(hw_size * 1.2) * 2 + 1  # a bit wider than 2 * hw_size, for latter offset convolution

    # Calculate the standard deviation from the FWHM
    offset_scaling = offset_degree / 2.6
    sigma = hw_size / (2 * np.sqrt(2 * np.log(2))) * (1 - 0.35 * offset_scaling)
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


    return skew_kernel


def generate_diff_kernels(hw_size, max_offset_degree=3.1):
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
            # TODO: recheck this process
            offset_degree = idx/(kernel_num-1) * max_offset_degree
            kernel = generate_skewed_gaussian_kernel_fitting_exp(hw_size, offset_degree=offset_degree)
            kernel_list.append(kernel)
    return np.array(kernel_list)

"""
the hw_size should be determined by the lr pixel distance relative to the hr pixel
for example, if the hr image is of shape [1024, 1024], and it is downsampled to [32, 32],
the `pixel_dist = 1024 / 32` = 32
the preferred `hw_size = int((pixel_dist / 5. * 8.9) / 2)`
"""