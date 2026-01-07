import numpy as np
import cv2

def image_left_to_right_rotate_pixel(img, rot_pix=1.):
    height, width = img.shape[:2]
    rot_pix = round(rot_pix) % width

    if len(img.shape) == 3:
        left_shifted_image = img[:, -rot_pix:width, :]
        right_shifted_image = img[:, 0:-rot_pix, :]
    else:
        left_shifted_image = img[:, -rot_pix:width]
        right_shifted_image = img[:, 0:-rot_pix]

    shifted_image = np.concatenate((left_shifted_image, right_shifted_image), axis=1)
    # cv2.imwrite('shifted_image.png', shifted_image)
    return shifted_image

def image_left_to_right_rotate_dgr(img, rot_dgr=1.):
    height, width = img.shape[:2]
    rot_pix = int(width * rot_dgr / 180.)

    shifted_image = image_left_to_right_rotate_pixel(img, rot_pix)
    # cv2.imwrite('shifted_image.png', shifted_image)
    return shifted_image

def generate_gaussian_kernel(hw_size, debug_flag=False, ns_small_kernel=False):
    '''
    generate a 2d gaussian kernel given with half-width-size
    :param hw_size: the half-width size of gaussian kernel in degree
    :param hemisphere: whether the image is a hemishpere or sphere
    :param debug_flag: to display or not
    :return: a 2d numpy array
    '''

    kernel_size = int(hw_size * 1.2) * 2 + 1  # a bit wider than 2 * hw_size, for latter offset convolution

    x = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
    y = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
    xx, yy = np.meshgrid(x, y)

    # Calculate the standard deviation from the FWHM
    if ns_small_kernel:
        sigma = hw_size / (2 * np.sqrt(2 * np.log(2))) / 3.5
    else:
        sigma = hw_size / (2 * np.sqrt(2 * np.log(2)))

    # Calculate the Gaussian kernel
    gaussian_kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    gaussian_kernel /= np.sum(gaussian_kernel) # (2 * np.pi * sigma ** 2)  # Normalize the kernel

    if debug_flag:
        print(np.sum(gaussian_kernel))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(gaussian_kernel)
        plt.show()

    return gaussian_kernel

def get_square_pos(size=50, mode="full"):
    '''
    generate the size*size positions [-1, 1]
    :param size: the number of column and row of the positions
    :return: [-1, 1] positions
    '''
    if mode == "full" or size > 200:
        x = np.linspace(-1., 1., size)
        y = np.linspace(-1., 1., size)
    else:
        x = np.arange(-1., 0.99, 2./size)
        y = np.arange(-1., 0.99, 2./size)
    X, Y = np.meshgrid(x, y)
    # Reshape X and Y into a single column for positions
    positions = np.column_stack((X.flatten(), Y.flatten()))
    omm_r = x[1] - x[0]

    return positions, omm_r

def img_convert_to_3channels(img_arr: np.ndarray, img_size: int):
    resized_img = cv2.resize(img_arr, (img_size, img_size))

    if len(resized_img.shape) < 3:
        resized_img = np.repeat(resized_img[:, :, np.newaxis], 3, axis=2)
    elif resized_img.shape[2] > 3:
        resized_img = resized_img[:, :, :3]

    return resized_img
