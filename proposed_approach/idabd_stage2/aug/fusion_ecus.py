# proposed_approach/aug/fusion_ecus.py
from dataclasses import dataclass, asdict
import numpy as np
import cv2

@dataclass
class FusionECUSConfig:
    # probabilities are handled by the dataset (fusion_p)
    alpha_unsharp: float = 0.7
    unsharp_sigma: float = 1.2
    unsharp_amount: float = 1.0

    alpha_contrast: float = 0.6
    clahe_clip: float = 2.0
    clahe_grid: int = 8

    alpha_edge: float = 0.6
    canny_t1: int = 50
    canny_t2: int = 150
    edge_dilate: int = 2

    def to_dict(self):
        d = asdict(self)
        d["clahe_grid"] = int(d["clahe_grid"])
        d["canny_t1"] = int(d["canny_t1"])
        d["canny_t2"] = int(d["canny_t2"])
        d["edge_dilate"] = int(d["edge_dilate"])
        return d


def _blend_bgr(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0:
        return a
    if alpha >= 1:
        return b
    return cv2.addWeighted(a, 1.0 - alpha, b, alpha, 0.0)


def _unsharp_bgr(img_bgr: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    sigma = float(max(0.01, sigma))
    amount = float(max(0.0, amount))
    blur = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharp = cv2.addWeighted(img_bgr, 1.0 + amount, blur, -amount, 0.0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def _clahe_contrast_bgr(img_bgr: np.ndarray, clip_limit: float, grid: int) -> np.ndarray:
    clip_limit = float(max(0.1, clip_limit))
    grid = int(max(2, grid))
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, A, B])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def _edge_overlay_bgr(
    img_bgr: np.ndarray, t1: int, t2: int, dilate: int, alpha_edge: float
) -> np.ndarray:
    t1 = int(max(1, t1))
    t2 = int(max(t1 + 1, t2))
    dilate = int(max(0, dilate))
    alpha_edge = float(max(0.0, alpha_edge))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(gray, t1, t2)

    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        e = cv2.dilate(e, k, iterations=1)

    e_bgr = cv2.cvtColor(e, cv2.COLOR_GRAY2BGR)
    out = cv2.addWeighted(img_bgr, 1.0, e_bgr, alpha_edge, 0.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def fusion_ecus_bgr(img_bgr: np.ndarray, cfg: FusionECUSConfig) -> np.ndarray:
    """
    Edge + Contrast + Unsharp fusion into a 3-ch BGR image.
    """
    u = _unsharp_bgr(img_bgr, sigma=cfg.unsharp_sigma, amount=cfg.unsharp_amount)
    m = _blend_bgr(img_bgr, u, cfg.alpha_unsharp)

    c = _clahe_contrast_bgr(m, clip_limit=cfg.clahe_clip, grid=cfg.clahe_grid)
    mc = _blend_bgr(m, c, cfg.alpha_contrast)

    e = _edge_overlay_bgr(
        mc, t1=cfg.canny_t1, t2=cfg.canny_t2, dilate=cfg.edge_dilate, alpha_edge=cfg.alpha_edge
    )
    return e