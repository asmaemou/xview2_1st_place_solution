from __future__ import annotations
from os import path
import torch
import torch.nn as nn

def strip_prefixes(sd: dict):
    prefixes = ["module.", "model.", "net."]
    out = {}
    for k, v in sd.items():
        kk = k
        for p in prefixes:
            if kk.startswith(p):
                kk = kk[len(p):]
        out[kk] = v
    return out

def load_state_dict_any(weight_path: str):
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

def build_damage_model_arch_from_weight_name(weight_path: str) -> nn.Module:
    import zoo.models as zm
    fname = path.basename(weight_path).lower()
    if fname.startswith("dpn92"):
        return zm.Dpn92_Unet_Double(pretrained=None)
    if fname.startswith("res34"):
        return zm.Res34_Unet_Double(pretrained=False)
    if fname.startswith("res50"):
        return zm.SeResNext50_Unet_Double(pretrained=None)
    if fname.startswith("se154"):
        return zm.SeNet154_Unet_Double(pretrained=None)
    raise ValueError(f"Unrecognized cls weight prefix: {fname}")

def build_damage_model_from_weight(weight_path: str) -> nn.Module:
    model = build_damage_model_arch_from_weight_name(weight_path)
    sd = load_state_dict_any(weight_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        print(f"[WARN] Unexpected keys in checkpoint (ignored): {len(unexpected)}")
    if missing:
        print(f"[WARN] Missing keys when loading checkpoint: {len(missing)}")
    return model

def build_loc_model_from_weight(weight_path: str) -> nn.Module:
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
