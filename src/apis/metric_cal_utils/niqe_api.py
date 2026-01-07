import torch
import pyiqa


def calculate_niqe(recon_lr_tensor: torch.Tensor, device=None) -> float:
    """
    计算输入 Tensor 的 NIQE 分数。

    Args:
        recon_lr_tensor: shape (1, h, w), 范围 [0, 1] 或 [0, 255]。
        device: 'cuda' 或 'cpu'。如果不指定，自动检测。

    Returns:
        float: NIQE 分数 (越低越好)。
    """
    # 1. 初始化设备
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 2. 模型单例缓存 (关键优化)
    # 检查函数下是否已经挂载了 model，如果没有则加载。
    # 这样多次调用该函数时，不会重复加载模型权重。
    if not hasattr(calculate_niqe, 'model'):
        print(f"Initializing NIQE model on {device}...")
        try:
            # as_loss=False 表示我们只想要原始分数
            calculate_niqe.model = pyiqa.create_metric('niqe', device=device, as_loss=False)
            calculate_niqe.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load pyiqa: {e}. Try: pip install pyiqa")

    # 3. 维度处理: (1, h, w) -> (1, 1, h, w)
    # PyIQA 要求输入为 (Batch, Channel, Height, Width)
    if recon_lr_tensor.dim() == 3:
        img_tensor = recon_lr_tensor.unsqueeze(0)
    elif recon_lr_tensor.dim() == 2:
        # 兼容 (h, w)
        img_tensor = recon_lr_tensor.unsqueeze(0).unsqueeze(0)
    else:
        img_tensor = recon_lr_tensor

    # 4. 数据范围与设备处理
    img_tensor = img_tensor.to(device)

    # 如果数据是 0-255，归一化到 0-1 (NIQE 敏感)
    if img_tensor.max() > 1.0 + 1e-6:
        img_tensor = img_tensor / 255.0

    # 5. 计算
    with torch.no_grad():
        score = calculate_niqe.model(img_tensor)

    return score.item()


# --- 测试代码 ---
if __name__ == "__main__":
    # 模拟输入 (1, 108, 108)
    dummy_img = torch.rand(1, 108, 108)

    # 第一次调用（会加载模型，稍慢）
    s1 = calculate_niqe(dummy_img)
    print(f"Score 1: {s1:.4f}")

    # 第二次调用（直接计算，极快）
    s2 = calculate_niqe(dummy_img)
    print(f"Score 2: {s2:.4f}")