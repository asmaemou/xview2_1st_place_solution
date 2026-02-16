import numpy as np
import cv2

def canny_edges_rgb(img_bgr: np.ndarray, t1: int = 50, t2: int = 150) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, int(t1), int(t2))
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def fuse_rgb_edges_bgr(
    img_bgr: np.ndarray,
    edge_w: float = 0.10,
    t1: int = 50,
    t2: int = 150,
) -> np.ndarray:
    """
    fused = (1-edge_w)*RGB + edge_w*EdgesRGB   (in BGR space)
    """
    edge_w = float(np.clip(edge_w, 0.0, 1.0))
    e = canny_edges_rgb(img_bgr, t1=t1, t2=t2).astype(np.float32)
    x = img_bgr.astype(np.float32)
    fused = (1.0 - edge_w) * x + edge_w * e
    return np.clip(fused, 0, 255).astype(np.uint8)

def fuse_rgb_edges_pair(
    pre_bgr: np.ndarray,
    post_bgr: np.ndarray,
    edge_w: float = 0.10,
    t1: int = 50,
    t2: int = 150,
):
    """
    Apply the SAME fusion rule to both pre and post.
    Returns (pre_fused, post_fused).
    """
    pre_f = fuse_rgb_edges_bgr(pre_bgr, edge_w=edge_w, t1=t1, t2=t2)
    post_f = fuse_rgb_edges_bgr(post_bgr, edge_w=edge_w, t1=t1, t2=t2)
    return pre_f, post_f