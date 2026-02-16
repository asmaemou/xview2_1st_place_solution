import numpy as np
import torch

IMG_DIR_DEFAULT = "../idabd/images"
MASK_DIR_DEFAULT = "../idabd/masks"
WEIGHTS_DIR_DEFAULT = "weights"
STAGE1_DIR_DEFAULT = "idabd_stage1_loc_ft_checkpoints"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IGNORE_LABEL = 255
CLASS_NAMES_4 = ["No-Damage", "Minor", "Major", "Destroyed"]

# Building pixels in IDABD mask convention (1..4)
BUILD_TENSOR_CPU = torch.tensor([1, 2, 3, 4], dtype=torch.long)

STAGE2_IN_CH_DEFAULT = 6
