import os
from os import path, makedirs, listdir
import sys
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

from zoo.models import SeResNext50_Unet_Double
from utils import preprocess_inputs

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# ---------
# CONFIG
# ---------
test_dir = 'idabd/images'     # ✅ changed from test/images
models_folder = 'weights'

if __name__ == '__main__':
    t0 = timeit.default_timer()

    seed = int(sys.argv[1])

    pred_folder = f'res50cls_cce_{seed}_tuned'
    makedirs(pred_folder, exist_ok=True)

    models = []

    # ✅ include .pth
    snap_to_load = f'res50_cls_cce_{seed}_tuned_best.pth'

    # ✅ IMPORTANT: avoid ImageNet download issues:
    # This assumes your zoo/models.py supports pretrained argument.
    # If it doesn't, leave as SeResNext50_Unet_Double() and we patch zoo/models.py.
    model = SeResNext50_Unet_Double(pretrained=None).cuda()
    model = nn.DataParallel(model).cuda()

    print(f"=> loading checkpoint '{snap_to_load}'")

    # ✅ PyTorch 2.6 fix
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
          .format(
              snap_to_load,
              checkpoint.get('epoch', 'NA'),
              checkpoint.get('best_score', 'NA')
          ))

    model.eval()
    models.append(model)

    with torch.no_grad():
        for f in tqdm(sorted(listdir(test_dir))):
            if '_pre_' not in f:
                continue

            fn = path.join(test_dir, f)

            img = cv2.imread(fn, cv2.IMREAD_COLOR)
            img2 = cv2.imread(fn.replace('_pre_', '_post_'), cv2.IMREAD_COLOR)

            img = np.concatenate([img, img2], axis=2)
            img = preprocess_inputs(img)

            # TTA
            inp = [
                img,
                img[::-1, ...],
                img[:, ::-1, ...],
                img[::-1, ::-1, ...]
            ]
            inp = np.asarray(inp, dtype='float')
            inp = torch.from_numpy(inp.transpose((0, 3, 1, 2))).float()
            inp = Variable(inp).cuda()

            pred = []
            for m in models:
                out = m(inp)
                out = torch.softmax(out, dim=1)
                out = out.detach().cpu().numpy()

                # keep same logic from original repo
                out[:, 0, ...] = 1 - out[:, 0, ...]

                pred.append(out[0, ...])
                pred.append(out[1, :, ::-1, :])
                pred.append(out[2, :, :, ::-1])
                pred.append(out[3, :, ::-1, ::-1])

            pred_full = np.asarray(pred).mean(axis=0)

            msk = (pred_full * 255).astype('uint8').transpose(1, 2, 0)

            # ✅ avoid .png.png
            out1 = f.replace('.png', '_part1.png')
            out2 = f.replace('.png', '_part2.png')

            cv2.imwrite(path.join(pred_folder, out1), msk[..., :3],
                        [cv2.IMWRITE_PNG_COMPRESSION, 9])
            cv2.imwrite(path.join(pred_folder, out2), msk[..., 2:],
                        [cv2.IMWRITE_PNG_COMPRESSION, 9])

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
