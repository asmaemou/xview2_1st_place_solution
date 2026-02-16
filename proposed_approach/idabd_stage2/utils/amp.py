from __future__ import annotations
from contextlib import contextmanager
import torch

@contextmanager
def autocast_disabled():
    # Works on any torch version; safe for float calibration ops.
    try:
        with torch.cuda.amp.autocast(enabled=False):
            yield
    except Exception:
        yield

def make_autocast(device: torch.device, enabled: bool):
    """
    Returns a context manager for autocast that works with both torch.amp and cuda.amp.
    """
    if enabled and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        def ctx():
            return torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda"))
        return ctx
    else:
        def ctx():
            return torch.cuda.amp.autocast(enabled=(device.type == "cuda" and enabled))
        return ctx

def make_grad_scaler(device: torch.device, enabled: bool):
    if device.type != "cuda":
        return torch.cuda.amp.GradScaler(enabled=False)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)
