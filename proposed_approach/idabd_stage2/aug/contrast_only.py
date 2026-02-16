import numpy as np
import cv2

def histeq_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """
    Histogram Equalization on luminance:
      BGR -> YCrCb, equalize Y, back to BGR
    """
    ycc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    ycc[:, :, 0] = cv2.equalizeHist(ycc[:, :, 0])
    out = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
    return out

def histeq_pair(pre_bgr: np.ndarray, post_bgr: np.ndarray):
    """
    Apply the same HistEq rule to both pre and post.
    """
    return histeq_bgr(pre_bgr), histeq_bgr(post_bgr)