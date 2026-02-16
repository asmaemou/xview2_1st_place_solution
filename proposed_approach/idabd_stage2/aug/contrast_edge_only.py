import numpy as np
import cv2

def clahe_contrast_bgr(img_bgr: np.ndarray, clip_limit: float = 2.0, grid: int = 8) -> np.ndarray:
    """CLAHE on L channel in LAB."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(grid), int(grid)))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

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
    if alpha <= 0.0:
        return a
    if alpha >= 1.0:
        return b
    return cv2.addWeighted(a, 1.0 - alpha, b, alpha, 0.0)

def overlay_edges_on_bgr(img_bgr: np.ndarray, edges_u8: np.ndarray, alpha_edge: float) -> np.ndarray:
    alpha = float(np.clip(alpha_edge, 0.0, 1.0))
    if alpha <= 0.0:
        return img_bgr
    edge_rgb = cv2.cvtColor(edges_u8, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(img_bgr, 1.0 - alpha, edge_rgb, alpha, 0.0)

def fuse_contrast_edge_post(
    post_bgr: np.ndarray,
    alpha_contrast: float = 0.45,
    alpha_edge: float = 0.10,
    clahe_clip: float = 2.0,
    clahe_grid: int = 8,
    canny_t1: int = 80,
    canny_t2: int = 200,
    edge_dilate: int = 0,
) -> np.ndarray:
    """
    POST-only fusion:
      1) CLAHE on LAB(L) -> post_c
      2) post_m = (1-alpha_contrast)*post + alpha_contrast*post_c
      3) Canny on post_m (+ optional dilation)
      4) Overlay edges onto post_m with alpha_edge
    """
    post_c = clahe_contrast_bgr(post_bgr, clip_limit=clahe_clip, grid=clahe_grid)
    post_m = blend_bgr(post_bgr, post_c, alpha_contrast)
    e = canny_edges(post_m, t1=canny_t1, t2=canny_t2, dilate=edge_dilate)
    post_f = overlay_edges_on_bgr(post_m, e, alpha_edge=alpha_edge)
    return post_f

def augment_pair_post_contrast_edge(
    pre_bgr: np.ndarray,
    post_bgr: np.ndarray,
    alpha_contrast: float = 0.45,
    alpha_edge: float = 0.10,
    clahe_clip: float = 2.0,
    clahe_grid: int = 8,
    canny_t1: int = 80,
    canny_t2: int = 200,
    edge_dilate: int = 0,
):
    """Keep PRE raw, apply fusion to POST only."""
    post_used = fuse_contrast_edge_post(
        post_bgr,
        alpha_contrast=alpha_contrast,
        alpha_edge=alpha_edge,
        clahe_clip=clahe_clip,
        clahe_grid=clahe_grid,
        canny_t1=canny_t1,
        canny_t2=canny_t2,
        edge_dilate=edge_dilate,
    )
    return pre_bgr, post_used