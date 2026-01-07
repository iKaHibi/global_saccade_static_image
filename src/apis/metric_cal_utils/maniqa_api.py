import os

# --- 必须放在任何其他 import 之前 ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import pyiqa


# ==========================================
# 1. BRISQUE (传统统计指标)
# ==========================================
def calculate_brisque(recon_lr_tensor: torch.Tensor, device=None) -> float:
    """
    计算 BRISQUE 分数。

    Args:
        recon_lr_tensor: shape (1, h, w), range [0, 1]
    Returns:
        float: 分数越低越好 (Lower is Better). 范围通常 [0, 100].
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 单例缓存模型
    if not hasattr(calculate_brisque, 'model'):
        print(f"Initializing BRISQUE model on {device}...")
        try:
            # as_loss=False: 返回原始分数
            calculate_brisque.model = pyiqa.create_metric('brisque', device=device, as_loss=False)
            calculate_brisque.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load BRISQUE: {e}")

    # 维度适配 (1, H, W) -> (1, 1, H, W)
    if recon_lr_tensor.dim() == 3:
        img_tensor = recon_lr_tensor.unsqueeze(0)
    elif recon_lr_tensor.dim() == 2:
        img_tensor = recon_lr_tensor.unsqueeze(0).unsqueeze(0)
    else:
        img_tensor = recon_lr_tensor

    img_tensor = img_tensor.to(device)

    # 归一化保障
    if img_tensor.max() > 1.0 + 1e-6:
        img_tensor = img_tensor / 255.0

    with torch.no_grad():
        score = calculate_brisque.model(img_tensor)

    return score.item()


# ==========================================
# 2. MANIQA (深度学习感知指标 - 推荐)
# ==========================================
def calculate_maniqa(recon_lr_tensor: torch.Tensor, device=None) -> float:
    """
    计算 MANIQA 分数 (Deep Learning based).

    Args:
        recon_lr_tensor: shape (1, h, w), range [0, 1]
    Returns:
        float: 分数越高越好 (Higher is Better).
               注意：这与 NIQE/BRISQUE 相反！通常范围在 [0, 1] 或更高。
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not hasattr(calculate_maniqa, 'model'):
        # print(f"Initializing MANIQA model on {device} (First run will download weights)...")
        try:
            # MANIQA 是 pyiqa 支持的 metric 之一
            calculate_maniqa.model = pyiqa.create_metric('maniqa', device=device, as_loss=False)
            calculate_maniqa.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load MANIQA: {e}. Try installing dependencies.")

    # 维度适配
    # MANIQA 通常期望 RGB 输入 (B, 3, H, W)。
    # 如果输入是单通道 (1, H, W)，我们需要复制通道维度 -> (1, 3, H, W)
    if recon_lr_tensor.dim() == 3:
        if recon_lr_tensor.size(0) == 1:
            img_tensor = recon_lr_tensor.repeat(1, 3, 1, 1)  # (1, H, W) -> (1, 3, H, W)
        else:
            img_tensor = recon_lr_tensor.unsqueeze(0)  # Assume (3, H, W) -> (1, 3, H, W)
    elif recon_lr_tensor.dim() == 2:
        img_tensor = recon_lr_tensor.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)  # (H, W) -> (1, 3, H, W)
    else:
        img_tensor = recon_lr_tensor

    img_tensor = img_tensor.to(device)

    if img_tensor.max() > 1.0 + 1e-6:
        img_tensor = img_tensor / 255.0

    with torch.no_grad():
        score = calculate_maniqa.model(img_tensor)

    return score.item()


# --- 测试代码 ---
if __name__ == "__main__":
    # 模拟一张 108x108 的图
    dummy = torch.rand(1, 108, 108)

    print("Testing BRISQUE...")
    s_brisque = calculate_brisque(dummy)
    print(f"BRISQUE: {s_brisque:.4f} (Lower is Better)")

    print("\nTesting MANIQA...")
    # MANIQA 需要下载模型，可能稍慢
    s_maniqa = calculate_maniqa(dummy)
    print(f"MANIQA:  {s_maniqa:.4f} (Higher is Better)")