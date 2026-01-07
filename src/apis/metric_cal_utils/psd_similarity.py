import torch
import torch.fft
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# --- get_radial_profile 函数保持不变 ---
def get_radial_profile(img_tensor: torch.Tensor):
    """
    计算图像的径向平均功率谱 (Radial Average Power Spectrum)。
    """
    img = img_tensor.float()
    h, w = img.shape[-2:]

    # 1. FFT & Shift
    f = torch.fft.fft2(img)
    fshift = torch.fft.fftshift(f)

    # 2. Power Spectrum
    magnitude = torch.abs(fshift) ** 2

    # 3. Coordinate Grid
    y, x = np.indices((h, w))
    center = np.array([h // 2, w // 2])
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    r = r.astype(int)

    # 4. Radial Average
    # move magnitude to cpu numpy
    mag_np = magnitude.detach().cpu().numpy()

    # 使用 bincount 计算每个半径环的总能量和像素数
    tbin = np.bincount(r.ravel(), mag_np.ravel())
    nr = np.bincount(r.ravel())

    epsilon = 1e-8
    # 避免除以零
    mask = nr > 0
    radial_profile = np.zeros_like(tbin, dtype=np.float64)
    radial_profile[mask] = tbin[mask] / nr[mask]

    # 只取到 Nyquist 频率 (最小边长的一半)
    # 确保不超过数组边界
    max_radius = min(min(h, w) // 2, len(radial_profile))
    return radial_profile[:max_radius]


def calculate_radial_psd_error(recon_frame: torch.Tensor, gt_frame: torch.Tensor,
                                      min_freq_threshold: int = 10, debug_flag: bool = True):
    """
    计算 Recon 和 GT 之间的径向对数频谱距离 (LSD)，并应用高通阈值。

    Args:
        recon_frame: (1, h, w)
        gt_frame: (1, H, W)
        min_freq_threshold: 计算误差时忽略的起始低频半径 (像素单位)。默认为 0。
        debug_flag: 是否绘制对比图。
    """
    h_lr, w_lr = recon_frame.shape[-2:]

    # min_freq_threshold = int(h_lr/9)

    # --- 1. GT Reference Preparation (Ideal Crop) ---
    # (代码与之前相同，省略注释)
    gt_fft = torch.fft.fft2(gt_frame, dim=(-2, -1))
    gt_shift = torch.fft.fftshift(gt_fft, dim=(-2, -1))

    c_h, c_w = gt_frame.shape[-2] // 2, gt_frame.shape[-1] // 2
    start_h = c_h - h_lr // 2
    start_w = c_w - w_lr // 2

    gt_crop_fft = gt_shift[..., start_h: start_h + h_lr, start_w: start_w + w_lr]

    gt_crop_ishift = torch.fft.ifftshift(gt_crop_fft, dim=(-2, -1))
    gt_ideal_lr = torch.fft.ifft2(gt_crop_ishift, dim=(-2, -1)).real

    # --- 2. Calculate Profiles ---
    profile_gt = get_radial_profile(gt_ideal_lr)
    profile_recon = get_radial_profile(recon_frame)

    # 确保两个 profile 长度一致 (处理可能的边缘情况)
    min_len = min(len(profile_gt), len(profile_recon))
    profile_gt = profile_gt[:min_len]
    profile_recon = profile_recon[:min_len]

    # --- 3. Convert to Log-Scale ---
    log_gt = np.log10(profile_gt + 1e-8)
    log_recon = np.log10(profile_recon + 1e-8)

    # --- 4. Calculate L1 Error with Masking ---
    # 计算全频段差异
    diff_full = np.abs(log_gt - log_recon)

    # 确保阈值不越界
    threshold_idx = min(min_freq_threshold, len(diff_full) - 1)
    threshold_idx = max(0, threshold_idx)  # 确保不小于0

    high_freq_gt = profile_gt[threshold_idx:]
    high_freq_recon = profile_recon[threshold_idx:]

    total_energy_gt = np.sum(high_freq_gt) + 1e-10
    total_energy_recon = np.sum(high_freq_recon) + 1e-10

    retention_rate = total_energy_recon / total_energy_gt

    # 【核心修改】只取阈值之后的部分计算平均值
    diff_high_freq = diff_full[threshold_idx:]

    if len(diff_high_freq) > 0:
        mean_error = diff_high_freq.mean()
    else:
        # 如果阈值设置得比最大半径还大，则没有数据用于计算
        mean_error = 0.0

    # --- 5. Debug Plotting (Updated) ---
    if debug_flag:
        plt.figure(figsize=(10, 6))
        freqs = np.arange(len(log_gt))

        # Plot lines
        plt.plot(freqs, log_gt, color='black', linewidth=2, linestyle='--', label='Reference (Ideal Sinc)')
        plt.plot(freqs, log_recon, color='red', linewidth=2, alpha=0.8, label='Algorithm Output')

        # Fill Error Gap (High Freq area)
        plt.fill_between(freqs[threshold_idx:],
                         log_gt[threshold_idx:],
                         log_recon[threshold_idx:],
                         color='red', alpha=0.2, label='High-Freq Error (Included)')

        # 【新增】可视化被屏蔽的低频区域
        if threshold_idx > 0:
            plt.axvline(x=threshold_idx, color='dimgray', linestyle='-', linewidth=1.5, label='Threshold boundary')
            # 用灰色填充被忽略的区域
            plt.axvspan(0, threshold_idx, color='gray', alpha=0.3, label='Low-Freq Ignored')

        # Update Title
        plt.title(f"Radial PSD Analysis (Threshold >= {threshold_idx} px)\nHigh-Frequency LSD Error: {mean_error:.4f}")

        plt.xlabel("Frequency Radius (pixels)")
        plt.ylabel("Log Power Magnitude")
        plt.legend(loc='upper right')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.xlim(left=0, right=len(freqs) - 1)
        plt.tight_layout()
        plt.show()

    return retention_rate


def calculate_spectral_metrics(recon_frame: torch.Tensor, gt_frame: torch.Tensor,
                               min_freq_threshold: int = 10, debug_flag=False):
    """
    计算两个标准频谱指标：
    1. LSD (Log-Spectral Distance): 衡量距离误差 (Lower is better)
    2. PCC (Pearson Correlation): 衡量形状一致性 (Higher is better)
    """
    h_lr, w_lr = recon_frame.shape[-2:]

    # 1. GT Reference (Ideal Crop)
    gt_fft = torch.fft.fft2(gt_frame, dim=(-2, -1))
    gt_shift = torch.fft.fftshift(gt_fft, dim=(-2, -1))
    c_h, c_w = gt_frame.shape[-2] // 2, gt_frame.shape[-1] // 2
    start_h, start_w = c_h - h_lr // 2, c_w - w_lr // 2
    gt_crop_fft = gt_shift[..., start_h: start_h + h_lr, start_w: start_w + w_lr]
    gt_crop_ishift = torch.fft.ifftshift(gt_crop_fft, dim=(-2, -1))
    gt_ideal_lr = torch.fft.ifft2(gt_crop_ishift, dim=(-2, -1)).real

    # 2. Profiles (Linear)
    profile_gt = get_radial_profile(gt_ideal_lr)
    profile_recon = get_radial_profile(recon_frame)
    min_len = min(len(profile_gt), len(profile_recon))

    # 3. High-pass Slicing & Log Conversion
    # 必须在 Log 域计算，符合感知和信号分贝特性
    threshold = min(min_freq_threshold, min_len - 2)#min(min_freq_threshold, min_len - 2)
    # threshold = min(int(min_len/32), min_len - 2)#min(min_freq_threshold, min_len - 2)

    # +1e-8 防止 log(0)
    log_gt = np.log10(profile_gt[:min_len] + 1e-8)[threshold:]
    log_recon = np.log10(profile_recon[:min_len] + 1e-8)[threshold:]

    # --- Metric 1: LSD (Log-Spectral Distance) ---
    # RMSE in Log domain
    lsd_score = np.sqrt(np.mean((log_gt - log_recon) ** 2))

    # --- Metric 2: PCC (Pearson Correlation) ---
    # 衡量形状是否一致
    if len(log_gt) > 2:
        pcc_score, _ = pearsonr(log_gt, log_recon)
    else:
        pcc_score = 0.0

    # --- 5. Debug Plotting (Updated) ---
    if debug_flag:
        plt.figure(figsize=(10, 6))
        threshold_idx = threshold
        freqs = np.linspace(0, 360, len(log_gt))

        # Plot lines
        plt.plot(freqs, log_gt, color='black', linewidth=2, linestyle='--', label='Reference (Ideal Sinc)')
        plt.plot(freqs, log_recon, color='red', linewidth=2, alpha=0.8, label='Algorithm Output')

        # 1. 强制设置 X 轴的显示范围为 0 到 360 (去除左右留白)
        plt.xlim(0, 360)
        # 2. (可选) 设置 X 轴的刻度间隔
        # 例如：每 45 度一个刻度，包含 0 和 360
        plt.xticks(np.arange(0, 361, 45))

        # Fill Error Gap (High Freq area)
        plt.fill_between(freqs[threshold_idx:],
                         log_gt[threshold_idx:],
                         log_recon[threshold_idx:],
                         color='red', alpha=0.2, label='High-Freq Error (Included)')

        # 【新增】可视化被屏蔽的低频区域
        if threshold_idx > 0:
            plt.axvline(x=threshold_idx, color='dimgray', linestyle='-', linewidth=1.5, label='Threshold boundary')
            # 用灰色填充被忽略的区域
            plt.axvspan(0, threshold_idx, color='gray', alpha=0.3, label='Low-Freq Ignored')

        # Update Title
        plt.title(f"Radial PSD Analysis (Threshold >= {threshold_idx} px)\nHigh-Frequency LSD Error: {lsd_score:.4f}")

        plt.xlabel("Frequency Radius (pixels)")
        plt.ylabel("Log Power Magnitude")
        plt.legend(loc='upper right')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        # plt.xlim(left=0, right=len(freqs) - 1)
        plt.tight_layout()
        plt.show()

    return lsd_score #pcc_score


def calculate_phase_similarity(recon_tensor: torch.Tensor, gt_tensor: torch.Tensor):
    """
    计算全频域相位相似度 (Global Phase Similarity)。
    自动处理尺寸不匹配问题：通过在频域裁剪 GT (Ideal Low-pass) 来与 LR 对齐。
    """
    # 确保输入是 float
    recon = recon_tensor.float()
    gt = gt_tensor.float()

    # 获取 LR 的目标尺寸
    h_lr, w_lr = recon.shape[-2:]

    # --- 1. 处理 GT (HR -> Ideal LR Spectrum) ---
    # 计算 GT 的 FFT 并 Shift
    fft_gt = torch.fft.fft2(gt, dim=(-2, -1))
    fft_gt_shift = torch.fft.fftshift(fft_gt, dim=(-2, -1))

    # 计算裁剪坐标 (Crop Center)
    h_hr, w_hr = gt.shape[-2:]
    c_h, c_w = h_hr // 2, w_hr // 2
    start_h = c_h - h_lr // 2
    start_w = c_w - w_lr // 2

    # 核心步骤：直接在频域截取 GT 的低频部分，使其尺寸 = LR
    fft_gt_crop = fft_gt_shift[..., start_h: start_h + h_lr, start_w: start_w + w_lr]

    # --- 2. 处理 Recon (LR) ---
    # LR 本身就是这个尺寸，直接 FFT + Shift 即可
    fft_recon = torch.fft.fft2(recon, dim=(-2, -1))
    fft_recon_shift = torch.fft.fftshift(fft_recon, dim=(-2, -1))

    # --- 3. 提取相位 (Unit Phase Vectors) ---
    # Phase = Z / |Z| (单位复数向量)
    # +1e-10 防止除以 0
    phase_gt = fft_gt_crop / (torch.abs(fft_gt_crop) + 1e-10)
    phase_recon = fft_recon_shift / (torch.abs(fft_recon_shift) + 1e-10)

    # --- 4. 计算相位相关性 ---
    # 公式: sum( cos(theta1 - theta2) )
    # 在复数域中，这等价于 sum( Real( z1 * conj(z2) ) )
    # 因为 z1 * conj(z2) = e^(i*t1) * e^(-i*t2) = e^(i*(t1-t2)) = cos(diff) + i*sin(diff)

    # 计算共轭相乘的实部
    dot_product = (phase_gt * torch.conj(phase_recon)).real

    # 求和
    correlation = torch.sum(dot_product)

    # --- 5. 归一化 ---
    # 理论最大值就是像素总数 (当所有相位完全相等时，cos(0)=1，sum = H*W)
    # 或者用严谨的向量范数归一化 (结果是一样的，因为都是单位向量)
    max_possible_score = h_lr * w_lr

    score = correlation / max_possible_score

    return score.item()

# --- 验证代码 ---
if __name__ == "__main__":
    # 设置随机种子以复现结果
    torch.manual_seed(42)
    np.random.seed(42)

    h_hr, w_hr = 1080, 1080
    h_lr, w_lr = 108, 108

    # 生成模拟数据: 1/f^1.5 噪声 (更接近自然图像统计，低频能量极高)
    # 这种数据最容易出现你提到的问题
    white = torch.randn(h_hr, w_hr)
    f_white = torch.fft.fft2(white)
    f_shift = torch.fft.fftshift(f_white)
    y, x = np.indices((h_hr, w_hr))
    center = np.array([h_hr // 2, w_hr // 2])
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2) + 1.0
    r = torch.from_numpy(r).float()
    # 使用更陡峭的斜率来模拟低频主导
    pink_fft = f_shift / (r ** 1.5)
    gt_tensor = torch.fft.ifft2(torch.fft.ifftshift(pink_fft)).real.unsqueeze(0)

    # 模拟 1: Good Algo (高频好，低频略差) - 用 Bicubic 模拟
    from torch.nn.functional import interpolate
    import torchvision.transforms.functional as TF

    recon_good = interpolate(gt_tensor.unsqueeze(0), size=(h_lr, w_lr), mode='bicubic', align_corners=False).squeeze(0)
    # 人为加一点点低频偏移噪声，模拟低频对不准的情况
    recon_good = recon_good * 1.02

    # 模拟 2: Bad Algo (高频差，低频好) - 强高斯模糊 + Bilinear
    recon_bad_pre = TF.gaussian_blur(gt_tensor, kernel_size=15, sigma=4.0)
    recon_bad = interpolate(recon_bad_pre.unsqueeze(0), size=(h_lr, w_lr), mode='bilinear',
                            align_corners=False).squeeze(0)

    THRESHOLD = 5
    print(f"Comparing with High-pass Threshold = {THRESHOLD} pixels...")

    print("Displaying Good Algorithm Plot...")
    # 我们预期这个误差现在应该更低
    err_good = calculate_radial_psd_error_masked(recon_good, gt_tensor, min_freq_threshold=THRESHOLD, debug_flag=True)

    print("Displaying Bad Algorithm Plot...")
    # 我们预期这个误差现在应该更高
    err_bad = calculate_radial_psd_error_masked(recon_bad, gt_tensor, min_freq_threshold=THRESHOLD, debug_flag=True)

    print(f"\n--- Results (Threshold >= {THRESHOLD}) ---")
    print(f"Algorithm 1 (Good High-Freq) Error: {err_good:.4f}")
    print(f"Algorithm 2 (Bad High-Freq)  Error: {err_bad:.4f}")

    if err_good < err_bad:
        print("\nSUCCESS: Metric now correctly favors Algorithm 1.")
    else:
        print("\nFAILURE: Metric still favors Algorithm 2 (adjust threshold or check data).")