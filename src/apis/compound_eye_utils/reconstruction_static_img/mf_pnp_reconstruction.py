from tqdm import tqdm
import os
import torch

from .utils_sisr import *
from .networks.network_unet import UNetRes as net

current_file_dir = os.path.dirname(os.path.abspath(__file__))

def todevice(x_list, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    return [img.to(device) for img in x_list]

def mf_pnp_reconstruction(lr_imgs, kernels, sf=4, K_max=3, initial_hr_estimation: torch.Tensor = None):
    """
    Performs Multi-Frame Plug-and-Play Super-Resolution.

    Args:
        lr_imgs (list[torch.Tensor]): List of low-resolution grayscale images.
        kernels (list[torch.Tensor]): List of filtering kernels.
        sf (int): The scaling factor.
        K_max (int): The total number of iterations.

    Returns:
        torch.Tensor: The final reconstructed high-resolution image.
    """
    # Previous Step: load tensors to device, load the denoiser model
    model_name = 'drunet_gray'
    n_channels = 1
    model_zoo = os.path.join(current_file_dir, "networks/pre_trained_weights")  # fixed
    model_path = os.path.join(model_zoo, model_name + '.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    
    lr_imgs = todevice(lr_imgs, device)
    kernels = todevice(kernels, device)

    model = net(in_nc=n_channels + 1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R',
                downsample_mode="strideconv", upsample_mode="convtranspose")
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    # Previous Step: get the mu & sigma (penalty term & noise level)
    noise_level_model = 3 / 255.
    modelSigma1 = 49  # set sigma_1, default: 49
    modelSigma2 = max(sf, noise_level_model * 255.)
    mu_list, sigma_list = get_rho_sigma(sigma=max(0.255 / 255., noise_level_model), iter_num=K_max,
                                         modelSigma1=modelSigma1, modelSigma2=modelSigma2, w=1)
    mu_list, sigma_list = torch.tensor(mu_list).to(device), torch.tensor(sigma_list).to(device)

    # 1. Pre-computation step
    print("--- Starting Pre-computation ---")
    z0, FBs, FBCs, FBFys = pre_compute_all_frames(lr_imgs, kernels, sf)
    z0 = z0.to(device)
    FBs = todevice(FBs, device)
    FBCs = todevice(FBCs, device)
    FBFys = todevice(FBFys, device)
    if initial_hr_estimation is None:
        z = z0
    else:
        z = initial_hr_estimation.to(device)
    print("Pre-computation finished.")

    # Pack the pre-computed variables into a dictionary for convenience
    args_dict = {"FBs": FBs, "FBCs": FBCs, "FBFys": FBFys, "sf": sf}

    # 2. Main Iteration Loop
    print(f"\n--- Starting {K_max} PnP Iterations ---")
    for i in tqdm(range(K_max)):
        # Step 1: Data Reconstruction
        args_dict["mu"] = mu_list[i]
        x = data_reconstruction_otf_api(z, args_dict)

        # Step 2: Denoising (Prior)
        z = denoise_step_api(x, model, sigma_list[i])
    
    print("PnP reconstruction finished.")
    return z