# proposed_approach/models/xview2_builders.py
from __future__ import annotations
from pathlib import Path
from os import path
import sys
import torch

def ensure_zoo_importable():
    """
    Make sure 'zoo' is importable, same logic you used before.
    """
    this_dir = Path(__file__).resolve().parent
    for cand in [this_dir, this_dir.parent, this_dir.parent.parent, this_dir.parent.parent.parent]:
        if (cand / "zoo").is_dir():
            sys.path.insert(0, str(cand))
            return

def strip_prefixes(sd: dict) -> dict:
    prefixes = ["module.", "model.", "net."]
    out = {}
    for k, v in sd.items():
        kk = k
        for p in prefixes:
            if kk.startswith(p):
                kk = kk[len(p):]
        out[kk] = v
    return out

def load_state_dict_any(weight_path: str) -> dict:
    try:
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(weight_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt

    if not isinstance(sd, dict):
        raise ValueError(f"Checkpoint not a state_dict mapping: {weight_path}")
    return strip_prefixes(sd)

def build_damage_model_from_weight(weight_path: str):
    ensure_zoo_importable()
    import zoo.models as zm

    fname = path.basename(weight_path).lower()
    if fname.startswith("dpn92"):
        model = zm.Dpn92_Unet_Double(pretrained=None)
    elif fname.startswith("res34"):
        model = zm.Res34_Unet_Double(pretrained=False)
    elif fname.startswith("res50"):
        model = zm.SeResNext50_Unet_Double(pretrained=None)
    elif fname.startswith("se154"):
        model = zm.SeNet154_Unet_Double(pretrained=None)
    else:
        raise ValueError(f"Unrecognized cls weight prefix: {fname}")

    sd = load_state_dict_any(weight_path)
    model.load_state_dict(sd, strict=True)
    return model

def build_loc_model_from_weight(weight_path: str):
    ensure_zoo_importable()
    import zoo.models as zm

    fname = path.basename(weight_path).lower()
    if fname.startswith("dpn92"):
        model = zm.Dpn92_Unet_Loc(pretrained=None)
    elif fname.startswith("res34"):
        model = zm.Res34_Unet_Loc(pretrained=False)
    elif fname.startswith("res50"):
        model = zm.SeResNext50_Unet_Loc(pretrained=None)
    elif fname.startswith("se154"):
        model = zm.SeNet154_Unet_Loc(pretrained=None)
    else:
        raise ValueError(f"Unrecognized localization weight prefix: {fname}")

    sd = load_state_dict_any(weight_path)
    model.load_state_dict(sd, strict=True)
    return model