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

from zoo.models import Dpn92_Unet_Double
from utils import preprocess_inputs

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# ---------
# CONFIG
# ---------
test_dir = 'idabd/images'   # ✅ changed from test/images
models_folder = 'weights'

if __name__ == '__main__':
    t0 = timeit.default_timer()

    seed = int(sys.argv[1])

    pred_folder = f'dpn92cls_cce_{seed}_tuned'
    makedirs(pred_folder, exist_ok=True)

    models = []

    # ✅ include .pth
    snap_to_load = f'dpn92_cls_cce_{seed}_tuned_best.pth'

    # ✅ CRITICAL: prevent downloading DPN92 imagenet weights (SSL crash)
    model = Dpn92_Unet_Double(pretrained=None).cuda()
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
                out = torch.softmax(out[:, :, ...], dim=1).detach().cpu().numpy()

                # match original repo logic
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
#########################
# import os
# from os import path, makedirs, listdir
# import sys
# import numpy as np
# np.random.seed(1)
# import random
# random.seed(1)

# import torch
# from torch import nn

# from tqdm import tqdm
# import timeit
# import cv2

# from zoo.models import Dpn92_Unet_Double
# from utils import preprocess_inputs

# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)

# # ---------
# # CONFIG
# # ---------
# test_dir = 'idabd/images'
# models_folder = 'weights'

# # AdaBN config (domain adaptation)
# DO_ADABN = True
# ADABN_MAX_IMAGES = 0   # 0 => use all _pre_ images; or set e.g. 80 for faster test
# ADABN_BATCH = 4        # batch size for BN adaptation pass
# USE_TTA_FOR_ADABN = False  # keep False (faster + more stable)

# def list_pre_files(img_dir):
#     files = [f for f in sorted(listdir(img_dir)) if ('_pre_' in f and f.lower().endswith('.png'))]
#     return files

# def load_pair_6ch(pre_path):
#     post_path = pre_path.replace('_pre_', '_post_')
#     img1 = cv2.imread(pre_path, cv2.IMREAD_COLOR)
#     img2 = cv2.imread(post_path, cv2.IMREAD_COLOR)
#     if img1 is None:
#         raise FileNotFoundError(f"Missing: {pre_path}")
#     if img2 is None:
#         raise FileNotFoundError(f"Missing: {post_path}")
#     img = np.concatenate([img1, img2], axis=2)
#     img = preprocess_inputs(img)
#     return img

# def to_tensor_bchw(img_list):
#     # img_list: list of HWC float images
#     arr = np.asarray(img_list, dtype='float32')
#     ten = torch.from_numpy(arr.transpose((0, 3, 1, 2))).float().cuda(non_blocking=True)
#     return ten

# def enable_bn_updates(model):
#     # Enable BN stat updates, but keep everything else in eval-like behavior
#     model.train()
#     for m in model.modules():
#         if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
#             m.train()  # update running_mean/var
#         else:
#             m.eval()   # freeze dropout etc.

# @torch.no_grad()
# def adabn_adapt(model, img_dir, max_images=0, batch_size=4):
#     pre_files = list_pre_files(img_dir)
#     if max_images and max_images > 0:
#         pre_files = pre_files[:max_images]

#     if len(pre_files) == 0:
#         print("[AdaBN] No _pre_ images found, skipping.")
#         return

#     print(f"[AdaBN] Adapting BN on {len(pre_files)} image-pairs (batch={batch_size}) ...")

#     enable_bn_updates(model)

#     # Warmup forward passes to update BN stats
#     buf = []
#     for f in tqdm(pre_files, desc="[AdaBN]"):
#         pre_path = path.join(img_dir, f)
#         img = load_pair_6ch(pre_path)

#         if USE_TTA_FOR_ADABN:
#             # Optional (slower). Usually not needed.
#             imgs = [img, img[::-1, ...], img[:, ::-1, ...], img[::-1, ::-1, ...]]
#             buf.extend(imgs)
#         else:
#             buf.append(img)

#         if len(buf) >= batch_size:
#             x = to_tensor_bchw(buf[:batch_size])
#             _ = model(x)  # forward only to update BN stats
#             buf = buf[batch_size:]

#     if len(buf) > 0:
#         x = to_tensor_bchw(buf)
#         _ = model(x)

#     model.eval()
#     print("[AdaBN] Done. Switched back to eval mode.")

# if __name__ == '__main__':
#     t0 = timeit.default_timer()

#     seed = int(sys.argv[1])

#     pred_folder = f'dpn92cls_cce_{seed}_tuned'
#     makedirs(pred_folder, exist_ok=True)

#     snap_to_load = f'dpn92_cls_cce_{seed}_tuned_best.pth'

#     # IMPORTANT: prevent downloading imagenet weights (SSL crash)
#     model = Dpn92_Unet_Double(pretrained=None).cuda()

#     print(f"=> loading checkpoint '{snap_to_load}'")
#     checkpoint = torch.load(
#         path.join(models_folder, snap_to_load),
#         map_location='cpu',
#         weights_only=False
#     )

#     loaded_dict = checkpoint['state_dict']
#     sd = model.state_dict()
#     for k in sd:
#         if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
#             sd[k] = loaded_dict[k]
#     model.load_state_dict(sd)

#     print("loaded checkpoint '{}' (epoch {}, best_score {})"
#           .format(snap_to_load,
#                   checkpoint.get('epoch', 'NA'),
#                   checkpoint.get('best_score', 'NA')))

#     model.eval()

#     # -------------------------
#     # AdaBN domain adaptation
#     # -------------------------
#     if DO_ADABN:
#         adabn_adapt(
#             model,
#             img_dir=test_dir,
#             max_images=ADABN_MAX_IMAGES,
#             batch_size=ADABN_BATCH
#         )

#     # -------------------------
#     # Inference (same as before)
#     # -------------------------
#     pre_files = list_pre_files(test_dir)

#     with torch.no_grad():
#         for f in tqdm(pre_files, desc="[Infer]"):
#             fn = path.join(test_dir, f)

#             img = load_pair_6ch(fn)

#             # TTA
#             inp = [
#                 img,
#                 img[::-1, ...],
#                 img[:, ::-1, ...],
#                 img[::-1, ::-1, ...]
#             ]
#             inp = to_tensor_bchw(inp)

#             out = model(inp)
#             out = torch.softmax(out, dim=1).detach().cpu().numpy()

#             # match original repo logic
#             out[:, 0, ...] = 1 - out[:, 0, ...]

#             pred = []
#             pred.append(out[0, ...])
#             pred.append(out[1, :, ::-1, :])
#             pred.append(out[2, :, :, ::-1])
#             pred.append(out[3, :, ::-1, ::-1])

#             pred_full = np.asarray(pred).mean(axis=0)
#             msk = (pred_full * 255).astype('uint8').transpose(1, 2, 0)

#             out1 = f.replace('.png', '_part1.png')
#             out2 = f.replace('.png', '_part2.png')

#             cv2.imwrite(path.join(pred_folder, out1), msk[..., :3],
#                         [cv2.IMWRITE_PNG_COMPRESSION, 9])
#             cv2.imwrite(path.join(pred_folder, out2), msk[..., 2:],
#                         [cv2.IMWRITE_PNG_COMPRESSION, 9])

#     elapsed = timeit.default_timer() - t0
#     print('Time: {:.3f} min'.format(elapsed / 60))

