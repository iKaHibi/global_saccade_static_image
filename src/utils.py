import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch

def save_json(data: Dict[str, Any], filepath: Path):
    """
    Saves a dictionary to a JSON file with human-readable formatting.

    Args:
        data (Dict[str, Any]): The dictionary data to save.
        filepath (Path): The path to the output JSON file.
    """
    print(f"Saving JSON data to {filepath}...")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print("Save complete.")

def load_json(filepath: Path) -> Dict[str, Any]:
    """Loads data from a JSON file."""
    print(f"Loading JSON data from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print("Load complete.")
    return data

# can add other common functions here later, like load_json, setup_logger, etc.
import numpy as np
import cv2
import os

def write_video(source_array_sequence: np.ndarray, src_video_path: str, fps=100.0):
    """
    Writes a NumPy array of grayscale frames to an MP4 video file.

    Args:
        source_array_sequence (np.ndarray): A NumPy array with shape
                                              [time_steps, height, width]
                                              containing the video frames.
                                              The values should be in a range
                                              that can be converted to uint8 (e.g., 0-255).
        src_video_path (str): The full path for the output .mp4 file.
        fps (float): The frame rate of the output video.
    """
    # --- 1. Input Validation and Preparation ---
    if not isinstance(source_array_sequence, np.ndarray):
        raise TypeError("Input 'source_array_sequence' must be a NumPy array.")

    if source_array_sequence.ndim != 3:
        raise ValueError("Input array must be 3-dimensional (time_steps, height, width).")

    # Get video dimensions from the array shape
    time_steps, height, width = source_array_sequence.shape

    # Ensure the output directory exists
    output_dir = os.path.dirname(src_video_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # --- 2. Define the Video Writer ---
    # Define the codec and create VideoWriter object. 'mp4v' is a common codec for .mp4 files.
    # Other options include 'XVID', 'MJPG', etc.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create a VideoWriter object. The frame size is (width, height).
    # The 'isColor' flag is set to False for grayscale video.
    out = cv2.VideoWriter(src_video_path, fourcc, fps, (width, height), isColor=False)

    if not out.isOpened():
        raise IOError(f"Could not open video writer for path: {src_video_path}")

    # --- 3. Write Frames to Video ---
    print(f"Writing {time_steps} frames to {src_video_path}...")
    try:
        source_arr_max_val = np.max(source_array_sequence)
        for i in range(time_steps):
            # Get the frame
            frame = source_array_sequence[i, :, :]

            # --- Data Normalization ---
            # Video writers expect frames as 8-bit unsigned integers (0-255).
            # This block handles conversion from other data types (like float 0-1).
            if frame.dtype != np.uint8:
                # If frame is float, scale from 0-1 to 0-255.
                if np.issubdtype(frame.dtype, np.floating) and source_arr_max_val < 2:
                    frame = (frame / source_arr_max_val * 255).astype(np.uint8)
                # Otherwise, just convert the type, assuming it's already in a suitable range.
                else:
                    frame = frame.astype(np.uint8)

            # Write the frame to the video file
            out.write(frame)

    finally:
        # --- 4. Release the Writer ---
        # This is crucial! It finalizes the video file.
        out.release()
        print("Video writing complete. File has been saved.")

def convert_array2tensor_list(arr, tensor_reshape=None, dtype=torch.float32):
    """
        Convert numpy array such that axis=0 becomes a list and other axes become torch tensors,
        with optional reshaping of each tensor.

        Args:
            arr: Input numpy array (e.g., shape [8, 32, 32])
            tensor_reshape: Optional target shape for each tensor (e.g., (16, 64))

        Returns:
            List of torch tensors
        """
    # Convert each slice to tensor and optionally reshape
    tensor_list = [torch.tensor(torch.from_numpy(x), dtype=dtype) for x in arr]

    if tensor_reshape is not None:
        tensor_list = [torch.tensor(t.reshape(*tensor_reshape), dtype=dtype) for t in tensor_list]

    return tensor_list


# ================ reconstruction related functions ================= #
def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

def save_image(img: np.ndarray, img_path: Path, resize_size: Optional[int] = None):
    """
    Validates, optionally resizes, and saves a NumPy array as an image file.

    Args:
        img (np.ndarray): The input image data as a NumPy array (dtype=uint8).
        img_path (Path): The path where the image will be saved.
        resize_size (Optional[int]): If provided, the image is resized to a
                                     square of (resize_size, resize_size).
                                     This takes precedence over other resizing.
    """
    # --- Validation Block ---
    if not isinstance(img, np.ndarray):
        raise TypeError("Input 'img' must be a NumPy array.")

    if img.dtype != np.uint8:
        raise ValueError(f"NumPy array must have dtype 'uint8', but got '{img.dtype}'.")
    
    # Squeeze to remove empty dimensions (e.g., from (1, H, W, 1) to (H, W))
    img = np.squeeze(img)

    # --- Resizing Logic ---
    # 2. If no specific size is given, check if the image is a line.
    if img.ndim == 1:
        size = int(np.sqrt(img.shape[0]))
        img = np.resize(img, (size, size))
    elif img.ndim >= 2:
        height, width = img.shape[:2]
        if height == 1 or width == 1:
            size = int(np.sqrt(height * width))
            # If image is a horizontal line, expand it to a square.
            # INTER_NEAREST is used to avoid creating artificial gradients.
            img = np.resize(img, (size, size))

    if resize_size is not None:
        # 1. If resize_size is specified, it takes priority.
        #    INTER_AREA is generally best for shrinking images (e.g., thumbnails).
        img = cv2.resize(img, (resize_size, resize_size), interpolation=cv2.INTER_AREA)


    # --- Saving Logic ---
    # Convert RGB to BGR for OpenCV, if it's a color image
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Ensure the parent directory for the image exists
    img_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the image to the specified path
    cv2.imwrite(str(img_path), img)
# def save_image(img: np.ndarray, img_path: Path):
#     if not isinstance(img, np.ndarray):
#         raise TypeError("Input 'image_data' must be a NumPy array.")

#     if img.dtype != np.uint8:
#         raise ValueError(f"NumPy array must have dtype 'uint8', but got '{img.dtype}'.")
#     # --- End of Validation Block ---

#     img = np.squeeze(img)
#     # change RGB to BGR, align with cv2
#     if img.ndim == 3:
#         img = img[:, :, [2, 1, 0]]
#     cv2.imwrite(img_path, img)


def todevice(x_list, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    return [img.to(device) for img in x_list]