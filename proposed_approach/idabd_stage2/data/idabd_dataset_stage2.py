from __future__ import annotations
from os import path
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from proposed_approach.aug.fusion_rgb import fusion_augment
from proposed_approach.data.idabd_preprocess import (
    pad_to_factor, preprocess_6ch, load_mask_raw
)

class IdaBDStage2Damage(Dataset):
    """
    Stage-2 dataset with destroyed-aware crop sampling + optional fusion augmentation.
    Returns:
      x: float tensor [6,H,W]
      y: long tensor [H,W] raw mask 0..4 or 255
      name: basename
    """
    def __init__(
        self,
        triplets,
        crop_size=512,
        is_train=True,
        fusion_p=0.75,
        min_build_px=50,
        seed=0,
        focus_destroyed_p=0.6,
        min_destroyed_px=50,
        crop_attempts=20,
    ):
        self.triplets = triplets
        self.crop_size = int(crop_size) if crop_size else 0
        self.is_train = is_train
        self.fusion_p = float(fusion_p)
        self.min_build_px = int(min_build_px)
        self.rng = random.Random(seed)

        self.focus_destroyed_p = float(focus_destroyed_p)
        self.min_destroyed_px = int(min_destroyed_px)
        self.crop_attempts = int(crop_attempts)

    def __len__(self):
        return len(self.triplets)

    def _random_crop(self, pre, post, mask, size):
        h, w = mask.shape[:2]
        if h < size or w < size:
            pad_h = max(0, size - h)
            pad_w = max(0, size - w)
            pre  = cv2.copyMakeBorder(pre,  0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            post = cv2.copyMakeBorder(post, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="edge")
            h, w = mask.shape[:2]

        want_destroyed = self.is_train and (self.rng.random() < self.focus_destroyed_p)

        for _ in range(self.crop_attempts):
            y0 = self.rng.randint(0, h - size)
            x0 = self.rng.randint(0, w - size)
            m = mask[y0:y0+size, x0:x0+size]

            if want_destroyed:
                dpx = int((m == 4).sum())
                if dpx >= self.min_destroyed_px:
                    return pre[y0:y0+size, x0:x0+size], post[y0:y0+size, x0:x0+size], m
            else:
                build_px = int(np.isin(m, [1, 2, 3, 4]).sum())
                if build_px >= self.min_build_px:
                    return pre[y0:y0+size, x0:x0+size], post[y0:y0+size, x0:x0+size], m

        for _ in range(self.crop_attempts):
            y0 = self.rng.randint(0, h - size)
            x0 = self.rng.randint(0, w - size)
            m = mask[y0:y0+size, x0:x0+size]
            build_px = int(np.isin(m, [1, 2, 3, 4]).sum())
            if build_px >= max(1, self.min_build_px // 2):
                return pre[y0:y0+size, x0:x0+size], post[y0:y0+size, x0:x0+size], m

        y0 = self.rng.randint(0, h - size)
        x0 = self.rng.randint(0, w - size)
        return (
            pre[y0:y0+size, x0:x0+size],
            post[y0:y0+size, x0:x0+size],
            mask[y0:y0+size, x0:x0+size],
        )

    def __getitem__(self, idx):
        pre_path, post_path, mask_path = self.triplets[idx]

        pre  = cv2.imread(pre_path, cv2.IMREAD_COLOR)
        post = cv2.imread(post_path, cv2.IMREAD_COLOR)
        if pre is None:
            raise FileNotFoundError(pre_path)
        if post is None:
            raise FileNotFoundError(post_path)

        m = load_mask_raw(mask_path)

        if self.is_train and (self.rng.random() < self.fusion_p):
            pre = fusion_augment(pre)
            post = fusion_augment(post)

        if self.is_train and self.crop_size > 0:
            pre, post, m = self._random_crop(pre, post, m, self.crop_size)

        pre, pad_hw = pad_to_factor(pre, 32)
        post, _ = pad_to_factor(post, 32)
        if pad_hw != (0, 0):
            m = np.pad(m, ((0, pad_hw[0]), (0, pad_hw[1])), mode="edge")

        x = preprocess_6ch(pre, post)
        y = torch.from_numpy(m).long()
        return x.float(), y.long(), path.basename(post_path)