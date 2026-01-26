import os
from os import path, makedirs, listdir
import numpy as np
np.random.seed(1)
import random
random.seed(1)

import torch
from torch import nn
from torch.autograd import Variable

from tqdm import tqdm
import timeit
import cv2

from zoo.models import SeNet154_Unet_Loc
from utils import preprocess_inputs

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# IDABD
test_dir = 'idabd/images'
pred_folder = 'pred154_loc'
models_folder = 'weights'

if __name__ == '__main__':
    t0 = timeit.default_timer()

    makedirs(pred_folder, exist_ok=True)

    models = []

    # 3-seed ensemble
    for seed in [0, 1, 2]:
        # IMPORTANT: in your weights folder you likely have .pth
        snap_to_load = f'se154_loc_{seed}_1_best.pth'

        # CRITICAL: pretrained=None => do NOT download ImageNet weights
        model = SeNet154_Unet_Loc(pretrained=None).cuda()
        model = nn.DataParallel(model).cuda()

        print(f"=> loading checkpoint '{snap_to_load}'")
        checkpoint = torch.load(
            path.join(models_folder, snap_to_load),
            map_location='cpu',
            weights_only=False
        )

        loaded_dict = checkpoint['state_dict']
        sd = model.state_dict()

        for k in sd:
            if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
                sd[k] = loaded_dict[k]

        model.load_state_dict(sd)

        print("loaded checkpoint '{}' (epoch {}, best_score {})"
              .format(snap_to_load,
                      checkpoint.get('epoch', 'NA'),
                      checkpoint.get('best_score', 'NA')))

        model.eval()
        models.append(model)

    with torch.no_grad():
        for f in tqdm(sorted(listdir(test_dir))):
            if '_pre_' not in f:
                continue

            fn = path.join(test_dir, f)

            img = cv2.imread(fn, cv2.IMREAD_COLOR)
            img = preprocess_inputs(img)

            # TTA: original, vflip, hflip, both
            inp = np.asarray([
                img,
                img[::-1, ...],
                img[:, ::-1, ...],
                img[::-1, ::-1, ...]
            ], dtype='float32')

            inp = torch.from_numpy(inp.transpose((0, 3, 1, 2))).float()
            inp = Variable(inp).cuda()

            pred = []
            for m in models:
                msk = m(inp)
                msk = torch.sigmoid(msk).detach().cpu().numpy()

                # invert transforms
                pred.append(msk[0, ...])
                pred.append(msk[1, :, ::-1, :])
                pred.append(msk[2, :, :, ::-1])
                pred.append(msk[3, :, ::-1, ::-1])

            pred_full = np.asarray(pred).mean(axis=0)

            msk_out = (pred_full * 255).astype('uint8').transpose(1, 2, 0)

            out1 = f.replace('.png', '_part1.png')  # no .png.png
            cv2.imwrite(path.join(pred_folder, out1), msk_out[..., 0],
                        [cv2.IMWRITE_PNG_COMPRESSION, 9])

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
