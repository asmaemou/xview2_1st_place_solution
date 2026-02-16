import numpy as np
import cv2

def unsharp_bgr(img_bgr: np.ndarray, sigma: float = 1.2, amount: float = 1.0) -> np.ndarray:
    sigma = float(max(0.01, sigma))
    amount = float(max(0.0, amount))
    blur = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharp = cv2.addWeighted(img_bgr, 1.0 + amount, blur, -amount, 0.0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def clahe_contrast_bgr(img_bgr: np.ndarray, clip_limit: float = 2.0, grid: int = 8) -> np.ndarray:
    """
    CLAHE on L channel in LAB (contrast enhancement without color blow-up).
    """
    clip_limit = float(max(0.1, clip_limit))
    grid = int(max(2, grid))

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, A, B])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return out

def blend_bgr(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0.0:
        return a
    if alpha >= 1.0:
        return b
    return cv2.addWeighted(a, 1.0 - alpha, b, alpha, 0.0)

def fuse_unsharp_contrast_post(
    post_bgr: np.ndarray,
    alpha_unsharp: float,
    unsharp_sigma: float,
    unsharp_amount: float,
    alpha_contrast: float,
    clahe_clip: float,
    clahe_grid: int,
) -> np.ndarray:
    # (A) unsharp then blend
    post_u = unsharp_bgr(post_bgr, sigma=unsharp_sigma, amount=unsharp_amount)
    post_m = blend_bgr(post_bgr, post_u, alpha_unsharp)

    # (B) CLAHE contrast then blend
    post_c = clahe_contrast_bgr(post_m, clip_limit=clahe_clip, grid=clahe_grid)
    post_f = blend_bgr(post_m, post_c, alpha_contrast)
    return post_f