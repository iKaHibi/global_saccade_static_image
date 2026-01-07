import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import tqdm

from src.apis.metric_cal_utils.cdf97 import *
import torch
import torch.nn.functional as F

to_cpu = lambda t: t.cpu() if t.is_cuda else t

def one_frame_FISH(frame_arr, alpha=0.8):
    def E_func(s):
        square_sum_mean = np.sum(np.square(s)) / s.size
        return np.log10(1 + square_sum_mean)

    # resize the frame
    width, height = frame_arr.shape
    # Calculate the new dimensions
    new_width = (width // 8) * 8
    new_height = (height // 8) * 8
    new_size = new_width if new_width > new_height else new_height
    # Resize the image to the nearest even dimensions
    im = np.resize(frame_arr, (new_size, new_size))
    im = np.array(im, dtype=float)

    m = im.tolist()
    m = fwt97_2d(m, 3)

    pix = np.zeros_like(im)

    seq_to_img(m, pix)  # Convert the list of lists matrix to an image.

    # get the En for 3 levels and adding it to fish_res
    fish_res = 0.
    En = np.zeros(3)
    for i in range(3):
        full_idx = int(new_size // (2 ** i))
        mid_idx = int(new_size // (2 ** (i+1)))
        LL = pix[:mid_idx, :mid_idx]
        LH = pix[mid_idx:full_idx, :mid_idx]
        HL = pix[:mid_idx, mid_idx:full_idx]
        HH = pix[mid_idx:full_idx, mid_idx:full_idx]

        E_LH = E_func(LH)
        E_HL = E_func(HL)
        E_HH = E_func(HH)

        En[i] = (1-alpha) * (E_LH + E_HL) / 2 + alpha * E_HH

        fish_res += 2**(3-(i+1)) * En[i]

    return fish_res

def calculate_fish_on_tensor(img_tensor):
    if len(img_tensor.shape) != 2:
        raise BaseException("Tensor Shape Should Be in 2 Dimensions")

    img_tensor = to_cpu(img_tensor)

    img_tensor = torch.clamp(img_tensor, 0, 1)  # Clamp the values to be within [0, 1]

    res = one_frame_FISH(img_tensor.numpy() * 255)
    return res

