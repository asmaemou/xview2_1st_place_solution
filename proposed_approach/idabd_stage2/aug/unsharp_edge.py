from __future__ import annotations
import cv2
import numpy as np

def unsharp_bgr(img_bgr: np.ndarray, sigma: float = 1.2, amount: float = 1.0) -> np.ndarray:
    sigma = float(max(0.01, sigma))
    amount = float(max(0.0, amount))
    blur = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharp = cv2.addWeighted(img_bgr, 1.0 + amount, blur, -amount, 0.0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def canny_edges(img_bgr: np.ndarray, t1: int = 80, t2: int = 200, dilate: int = 0) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(gray, int(t1), int(t2))
    if dilate and int(dilate) > 0:
        k = int(dilate) * 2 + 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        e = cv2.dilate(e, ker, iterations=1)
    return e

def blend_bgr(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0: return a
    if alpha >= 1: return b
    return cv2.addWeighted(a, 1.0 - alpha, b, alpha, 0.0)

def overlay_edges_on_bgr(img_bgr: np.ndarray, edges_u8: np.ndarray, alpha_edge: float) -> np.ndarray:
    alpha = float(np.clip(alpha_edge, 0.0, 1.0))
    if alpha <= 0: return img_bgr
    edge_rgb = cv2.cvtColor(edges_u8, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(img_bgr, 1.0 - alpha, edge_rgb, alpha, 0.0)

def fuse_unsharp_edge_post(
    post_bgr: np.ndarray,
    alpha_unsharp: float,
    unsharp_sigma: float,
    unsharp_amount: float,
    alpha_edge: float,
    canny_t1: int,
    canny_t2: int,
    edge_dilate: int,
) -> np.ndarray:
    post_u = unsharp_bgr(post_bgr, sigma=unsharp_sigma, amount=unsharp_amount)
    post_m = blend_bgr(post_bgr, post_u, alpha_unsharp)
    e = canny_edges(post_m, t1=canny_t1, t2=canny_t2, dilate=edge_dilate)
    return overlay_edges_on_bgr(post_m, e, alpha_edge=alpha_edge)