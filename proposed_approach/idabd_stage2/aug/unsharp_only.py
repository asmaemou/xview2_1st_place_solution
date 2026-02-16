from __future__ import annotations
import cv2
import numpy as np

def unsharp_bgr(img_bgr: np.ndarray, amount: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    if amount <= 0:
        return img_bgr
    blur = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    sharp = cv2.addWeighted(img_bgr, 1.0 + float(amount), blur, -float(amount), 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def blend_bgr(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0: return a
    if alpha >= 1: return b
    return cv2.addWeighted(a, 1.0 - alpha, b, alpha, 0.0)

def post_unsharp_only(post_bgr: np.ndarray, alpha: float, amount: float, sigma: float) -> np.ndarray:
    u = unsharp_bgr(post_bgr, amount=amount, sigma=sigma)
    return blend_bgr(post_bgr, u, alpha=alpha)
