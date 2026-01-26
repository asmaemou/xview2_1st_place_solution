# import os
# os.environ["MKL_NUM_THREADS"] = "2"
# os.environ["NUMEXPR_NUM_THREADS"] = "2"
# os.environ["OMP_NUM_THREADS"] = "2"

# from os import path, makedirs, listdir
# import sys
# import numpy as np
# np.random.seed(1)
# import random
# random.seed(1)

# # FDA
# from glob import glob
# from fda import apply_fda_uint8

# import torch
# from torch import nn
# from torch.backends import cudnn
# from torch.utils.data import Dataset, DataLoader
# import torch.optim.lr_scheduler as lr_scheduler

# from torch.cuda.amp import autocast, GradScaler

# from adamw import AdamW
# from losses import dice_round, ComboLoss

# from tqdm import tqdm
# import timeit
# import cv2

# from zoo.models import SeNet154_Unet_Double
# from imgaug import augmenters as iaa
# from utils import *

# from skimage.morphology import square, dilation
# from sklearn.model_selection import train_test_split
# import gc

# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)

# # =========================
# # DATA CONFIG
# # =========================
# # Supervised fine-tune on IDABD:
# train_dirs = ['idabd']

# models_folder = 'weights'
# input_shape = (320, 320)

# # =========================
# # FDA DOMAIN ADAPTATION
# # =========================
# USE_FDA = True        # master switch
# FDA_PROB = 0.5          # probability per sample
# FDA_BETA = 0.01         # strength

# IDABD_STYLE_DIR = "idabd/images"
# IDABD_STYLE_FILES = sorted(glob(path.join(IDABD_STYLE_DIR, "*_pre_disaster.png")))
# # NOTE: do NOT print here (Windows DataLoader workers re-import the file)

# # =========================
# # PATH HELPERS (safe)
# # =========================
# def img_to_mask_path(fn: str) -> str:
#     # .../<dataset>/images/<name>.png  ->  .../<dataset>/masks/<name>.png
#     root = path.dirname(path.dirname(fn))
#     return path.join(root, "masks", path.basename(fn))

# def pre_to_post(fn: str) -> str:
#     return fn.replace('_pre_disaster', '_post_disaster')

# # =========================
# # COLLECT FILES
# # =========================
# all_files = []
# for d in train_dirs:
#     img_dir = path.join(d, 'images')
#     for f in sorted(listdir(img_dir)):
#         if f.endswith('.png') and ('_pre_disaster.png' in f):
#             all_files.append(path.join(img_dir, f))

# if len(all_files) == 0:
#     raise RuntimeError(f"No *_pre_disaster.png files found under: {train_dirs} (expected <dir>/images/*.png)")

# # =========================
# # DATASETS
# # =========================
# class TrainData(Dataset):
#     def __init__(self, train_idxs):
#         super().__init__()
#         self.train_idxs = train_idxs
#         self.elastic = iaa.ElasticTransformation(alpha=(0.25, 1.2), sigma=0.2)
#         self.style_files = IDABD_STYLE_FILES

#     def __len__(self):
#         return len(self.train_idxs)

#     def __getitem__(self, idx):
#         _idx = self.train_idxs[idx]
#         fn = all_files[_idx]

#         img = cv2.imread(fn, cv2.IMREAD_COLOR)
#         img2 = cv2.imread(pre_to_post(fn), cv2.IMREAD_COLOR)
#         if img is None or img2 is None:
#             raise FileNotFoundError(f"Could not read image pair: {fn} / {pre_to_post(fn)}")

#         msk0_path = img_to_mask_path(fn)
#         msk1_path = img_to_mask_path(pre_to_post(fn))

#         msk0 = cv2.imread(msk0_path, cv2.IMREAD_UNCHANGED)
#         lbl_msk1 = cv2.imread(msk1_path, cv2.IMREAD_UNCHANGED)
#         if msk0 is None or lbl_msk1 is None:
#             raise FileNotFoundError(f"Could not read mask pair: {msk0_path} / {msk1_path}")

#         msk1 = np.zeros_like(lbl_msk1)
#         msk2 = np.zeros_like(lbl_msk1)
#         msk3 = np.zeros_like(lbl_msk1)
#         msk4 = np.zeros_like(lbl_msk1)
#         msk1[lbl_msk1 == 1] = 255
#         msk2[lbl_msk1 == 2] = 255
#         msk3[lbl_msk1 == 3] = 255
#         msk4[lbl_msk1 == 4] = 255

#         # flips
#         if random.random() > 0.5:
#             img = img[::-1, ...]
#             img2 = img2[::-1, ...]
#             msk0 = msk0[::-1, ...]
#             msk1 = msk1[::-1, ...]
#             msk2 = msk2[::-1, ...]
#             msk3 = msk3[::-1, ...]
#             msk4 = msk4[::-1, ...]

#         # rotations
#         if random.random() > 0.0001:
#             rot = random.randrange(4)
#             if rot > 0:
#                 img = np.rot90(img, k=rot)
#                 img2 = np.rot90(img2, k=rot)
#                 msk0 = np.rot90(msk0, k=rot)
#                 msk1 = np.rot90(msk1, k=rot)
#                 msk2 = np.rot90(msk2, k=rot)
#                 msk3 = np.rot90(msk3, k=rot)
#                 msk4 = np.rot90(msk4, k=rot)

#         # shifts
#         if random.random() > 0.5:
#             shift_pnt = (random.randint(-320, 320), random.randint(-320, 320))
#             img = shift_image(img, shift_pnt)
#             img2 = shift_image(img2, shift_pnt)
#             msk0 = shift_image(msk0, shift_pnt)
#             msk1 = shift_image(msk1, shift_pnt)
#             msk2 = shift_image(msk2, shift_pnt)
#             msk3 = shift_image(msk3, shift_pnt)
#             msk4 = shift_image(msk4, shift_pnt)

#         # rotate+scale
#         if random.random() > 0.05:
#             rot_pnt = (img.shape[0] // 2 + random.randint(-320, 320),
#                        img.shape[1] // 2 + random.randint(-320, 320))
#             scale = 0.9 + random.random() * 0.2
#             angle = random.randint(0, 20) - 10
#             if (angle != 0) or (scale != 1):
#                 img = rotate_image(img, angle, scale, rot_pnt)
#                 img2 = rotate_image(img2, angle, scale, rot_pnt)
#                 msk0 = rotate_image(msk0, angle, scale, rot_pnt)
#                 msk1 = rotate_image(msk1, angle, scale, rot_pnt)
#                 msk2 = rotate_image(msk2, angle, scale, rot_pnt)
#                 msk3 = rotate_image(msk3, angle, scale, rot_pnt)
#                 msk4 = rotate_image(msk4, angle, scale, rot_pnt)

#         # crop
#         crop_size = input_shape[0]
#         if random.random() > 0.05:
#             crop_size = random.randint(int(input_shape[0] / 1.15), int(input_shape[0] / 0.85))

#         bst_x0 = random.randint(0, img.shape[1] - crop_size)
#         bst_y0 = random.randint(0, img.shape[0] - crop_size)
#         bst_sc = -1
#         try_cnt = random.randint(1, 10)
#         for _ in range(try_cnt):
#             x0 = random.randint(0, img.shape[1] - crop_size)
#             y0 = random.randint(0, img.shape[0] - crop_size)
#             _sc = (msk2[y0:y0+crop_size, x0:x0+crop_size].sum() * 5 +
#                    msk3[y0:y0+crop_size, x0:x0+crop_size].sum() * 5 +
#                    msk4[y0:y0+crop_size, x0:x0+crop_size].sum() * 2 +
#                    msk1[y0:y0+crop_size, x0:x0+crop_size].sum())
#             if _sc > bst_sc:
#                 bst_sc = _sc
#                 bst_x0 = x0
#                 bst_y0 = y0

#         x0, y0 = bst_x0, bst_y0
#         img = img[y0:y0+crop_size, x0:x0+crop_size, :]
#         img2 = img2[y0:y0+crop_size, x0:x0+crop_size, :]
#         msk0 = msk0[y0:y0+crop_size, x0:x0+crop_size]
#         msk1 = msk1[y0:y0+crop_size, x0:x0+crop_size]
#         msk2 = msk2[y0:y0+crop_size, x0:x0+crop_size]
#         msk3 = msk3[y0:y0+crop_size, x0:x0+crop_size]
#         msk4 = msk4[y0:y0+crop_size, x0:x0+crop_size]

#         if crop_size != input_shape[0]:
#             img = cv2.resize(img, input_shape, interpolation=cv2.INTER_LINEAR)
#             img2 = cv2.resize(img2, input_shape, interpolation=cv2.INTER_LINEAR)
#             msk0 = cv2.resize(msk0, input_shape, interpolation=cv2.INTER_LINEAR)
#             msk1 = cv2.resize(msk1, input_shape, interpolation=cv2.INTER_LINEAR)
#             msk2 = cv2.resize(msk2, input_shape, interpolation=cv2.INTER_LINEAR)
#             msk3 = cv2.resize(msk3, input_shape, interpolation=cv2.INTER_LINEAR)
#             msk4 = cv2.resize(msk4, input_shape, interpolation=cv2.INTER_LINEAR)

#         # FDA (same style for both pre/post)
#         if USE_FDA and self.style_files and (random.random() < FDA_PROB):
#             tgt_fn = random.choice(self.style_files)
#             tgt = cv2.imread(tgt_fn, cv2.IMREAD_COLOR)
#             if tgt is not None:
#                 tgt = cv2.resize(tgt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
#                 img = apply_fda_uint8(img, tgt, beta=FDA_BETA)
#                 img2 = apply_fda_uint8(img2, tgt, beta=FDA_BETA)

#         # color augs
#         if random.random() > 0.9:
#             img = shift_channels(img, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))
#         elif random.random() > 0.9:
#             img2 = shift_channels(img2, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))

#         if random.random() > 0.9:
#             img = change_hsv(img, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))
#         elif random.random() > 0.9:
#             img2 = change_hsv(img2, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))

#         if random.random() > 0.9:
#             if random.random() > 0.9:
#                 img = clahe(img)
#             elif random.random() > 0.9:
#                 img = gauss_noise(img)
#             elif random.random() > 0.9:
#                 img = cv2.blur(img, (3, 3))
#         elif random.random() > 0.9:
#             if random.random() > 0.9:
#                 img = saturation(img, 0.9 + random.random() * 0.2)
#             elif random.random() > 0.9:
#                 img = brightness(img, 0.9 + random.random() * 0.2)
#             elif random.random() > 0.9:
#                 img = contrast(img, 0.9 + random.random() * 0.2)

#         if random.random() > 0.9:
#             if random.random() > 0.9:
#                 img2 = clahe(img2)
#             elif random.random() > 0.9:
#                 img2 = gauss_noise(img2)
#             elif random.random() > 0.9:
#                 img2 = cv2.blur(img2, (3, 3))
#         elif random.random() > 0.9:
#             if random.random() > 0.9:
#                 img2 = saturation(img2, 0.9 + random.random() * 0.2)
#             elif random.random() > 0.9:
#                 img2 = brightness(img2, 0.9 + random.random() * 0.2)
#             elif random.random() > 0.9:
#                 img2 = contrast(img2, 0.9 + random.random() * 0.2)

#         if random.random() > 0.9:
#             el_det = self.elastic.to_deterministic()
#             img = el_det.augment_image(img)

#         if random.random() > 0.9:
#             el_det = self.elastic.to_deterministic()
#             img2 = el_det.augment_image(img2)

#         # masks to channels
#         msk0 = msk0[..., np.newaxis]
#         msk1 = msk1[..., np.newaxis]
#         msk2 = msk2[..., np.newaxis]
#         msk3 = msk3[..., np.newaxis]
#         msk4 = msk4[..., np.newaxis]

#         msk = np.concatenate([msk0, msk1, msk2, msk3, msk4], axis=2)
#         msk = (msk > 127)

#         # winner-style mask logic
#         msk[..., 0] = True
#         msk[..., 1] = dilation(msk[..., 1], square(5))
#         msk[..., 2] = dilation(msk[..., 2], square(5))
#         msk[..., 3] = dilation(msk[..., 3], square(5))
#         msk[..., 4] = dilation(msk[..., 4], square(5))
#         msk[..., 1][msk[..., 2:].max(axis=2)] = False
#         msk[..., 3][msk[..., 2]] = False
#         msk[..., 4][msk[..., 2]] = False
#         msk[..., 4][msk[..., 3]] = False
#         msk[..., 0][msk[..., 1:].max(axis=2)] = False
#         msk = msk * 1

#         lbl_msk = msk.argmax(axis=2)

#         img = np.concatenate([img, img2], axis=2)
#         img = preprocess_inputs(img)

#         img = torch.from_numpy(img.transpose((2, 0, 1))).float()
#         msk = torch.from_numpy(msk.transpose((2, 0, 1))).long()

#         return {'img': img, 'msk': msk, 'lbl_msk': lbl_msk, 'fn': fn}


# class ValData(Dataset):
#     def __init__(self, image_idxs):
#         super().__init__()
#         self.image_idxs = image_idxs

#     def __len__(self):
#         return len(self.image_idxs)

#     def __getitem__(self, idx):
#         _idx = self.image_idxs[idx]
#         fn = all_files[_idx]

#         img = cv2.imread(fn, cv2.IMREAD_COLOR)
#         img2 = cv2.imread(pre_to_post(fn), cv2.IMREAD_COLOR)
#         if img is None or img2 is None:
#             raise FileNotFoundError(f"Could not read image pair: {fn} / {pre_to_post(fn)}")

#         msk0_path = img_to_mask_path(fn)
#         msk1_path = img_to_mask_path(pre_to_post(fn))

#         msk0 = cv2.imread(msk0_path, cv2.IMREAD_UNCHANGED)
#         lbl_msk1 = cv2.imread(msk1_path, cv2.IMREAD_UNCHANGED)
#         if msk0 is None or lbl_msk1 is None:
#             raise FileNotFoundError(f"Could not read mask pair: {msk0_path} / {msk1_path}")

#         # IDABD validation: use GT buildings as "loc mask" (no stage-1 dependency)
#         msk_loc = (msk0 > 0)

#         msk1 = np.zeros_like(lbl_msk1)
#         msk2 = np.zeros_like(lbl_msk1)
#         msk3 = np.zeros_like(lbl_msk1)
#         msk4 = np.zeros_like(lbl_msk1)
#         msk1[lbl_msk1 == 1] = 255
#         msk2[lbl_msk1 == 2] = 255
#         msk3[lbl_msk1 == 3] = 255
#         msk4[lbl_msk1 == 4] = 255

#         msk0 = msk0[..., np.newaxis]
#         msk1 = msk1[..., np.newaxis]
#         msk2 = msk2[..., np.newaxis]
#         msk3 = msk3[..., np.newaxis]
#         msk4 = msk4[..., np.newaxis]

#         msk = np.concatenate([msk0, msk1, msk2, msk3, msk4], axis=2)
#         msk = (msk > 127)
#         msk = msk * 1

#         lbl_msk = msk[..., 1:].argmax(axis=2)

#         img = np.concatenate([img, img2], axis=2)
#         img = preprocess_inputs(img)

#         img = torch.from_numpy(img.transpose((2, 0, 1))).float()
#         msk = torch.from_numpy(msk.transpose((2, 0, 1))).long()

#         return {'img': img, 'msk': msk, 'lbl_msk': lbl_msk, 'fn': fn, 'msk_loc': msk_loc}


# # =========================
# # VALIDATION
# # =========================
# def validate(net, data_loader):
#     tp = np.zeros((5,))
#     fp = np.zeros((5,))
#     fn = np.zeros((5,))

#     _thr = 0.3

#     with torch.no_grad():
#         for sample in tqdm(data_loader):
#             msks = sample["msk"].numpy()
#             lbl_msk = sample["lbl_msk"].numpy()
#             imgs = sample["img"].cuda(non_blocking=True)
#             msk_loc = sample["msk_loc"].numpy() * 1

#             out = net(imgs)

#             # msk_pred = msk_loc
#             loc_prob = torch.sigmoid(out[:, 0]).cpu().numpy()
#             msk_pred = (loc_prob > 0.3).astype(np.uint8)

#             msk_damage_pred = torch.softmax(out, dim=1).cpu().numpy()[:, 1:, ...]

#             for j in range(msks.shape[0]):
#                 tp[4] += np.logical_and(msks[j, 0] > 0, msk_pred[j] > 0).sum()
#                 fn[4] += np.logical_and(msks[j, 0] < 1, msk_pred[j] > 0).sum()
#                 fp[4] += np.logical_and(msks[j, 0] > 0, msk_pred[j] < 1).sum()

#                 targ = lbl_msk[j][msks[j, 0] > 0]
#                 pred = msk_damage_pred[j].argmax(axis=0)
#                 pred = pred * (msk_pred[j] > _thr)
#                 pred = pred[msks[j, 0] > 0]
#                 for c in range(4):
#                     tp[c] += np.logical_and(pred == c, targ == c).sum()
#                     fn[c] += np.logical_and(pred != c, targ == c).sum()
#                     fp[c] += np.logical_and(pred == c, targ != c).sum()

#     d0 = 2 * tp[4] / (2 * tp[4] + fp[4] + fn[4] + 1e-9)

#     f1_sc = np.zeros((4,))
#     for c in range(4):
#         f1_sc[c] = 2 * tp[c] / (2 * tp[c] + fp[c] + fn[c] + 1e-9)

#     f1 = 4 / np.sum(1.0 / (f1_sc + 1e-6))

#     sc = 0.3 * d0 + 0.7 * f1
#     print("Val Score: {}, Dice: {}, F1: {}, F1_0: {}, F1_1: {}, F1_2: {}, F1_3: {}".format(
#         sc, d0, f1, f1_sc[0], f1_sc[1], f1_sc[2], f1_sc[3]
#     ))
#     return sc


# def evaluate_val(data_val, best_score, model, snapshot_name, current_epoch):
#     model = model.eval()
#     d = validate(model, data_loader=data_val)

#     if d > best_score:
#         save_path = path.join(models_folder, snapshot_name + '_best.pth')
#         torch.save({
#             'epoch': current_epoch + 1,
#             'state_dict': model.state_dict(),
#             'best_score': d,
#         }, save_path)
#         best_score = d
#         print(f"[SAVE] {save_path}")

#     print("score: {}\tscore_best: {}".format(d, best_score))
#     return best_score


# # =========================
# # TRAIN
# # =========================
# def train_epoch(current_epoch, seg_loss, ce_loss, model, optimizer, scheduler, train_data_loader, scaler):
#     losses = AverageMeter()
#     losses1 = AverageMeter()
#     dices = AverageMeter()

#     iterator = tqdm(train_data_loader)
#     model.train()

#     for sample in iterator:
#         imgs = sample["img"].cuda(non_blocking=True)
#         msks = sample["msk"].cuda(non_blocking=True)
#         lbl_msk = sample["lbl_msk"].cuda(non_blocking=True)

#         optimizer.zero_grad(set_to_none=True)

#         with autocast():
#             out = model(imgs)

#             loss0 = seg_loss(out[:, 0, ...], msks[:, 0, ...])
#             loss1 = seg_loss(out[:, 1, ...], msks[:, 1, ...])
#             loss2 = seg_loss(out[:, 2, ...], msks[:, 2, ...])
#             loss3 = seg_loss(out[:, 3, ...], msks[:, 3, ...])
#             loss4 = seg_loss(out[:, 4, ...], msks[:, 4, ...])
#             loss5 = ce_loss(out, lbl_msk)

#             loss = 0.1 * loss0 + 0.1 * loss1 + 0.6 * loss2 + 0.3 * loss3 + 0.2 * loss4 + loss5 * 8

#         with torch.no_grad():
#             _probs = 1 - torch.sigmoid(out[:, 0, ...].float())
#             dice_sc = 1 - dice_round(_probs, 1 - msks[:, 0, ...].float())

#         losses.update(loss.item(), imgs.size(0))
#         losses1.update(loss5.item(), imgs.size(0))
#         dices.update(dice_sc, imgs.size(0))

#         scaler.scale(loss).backward()
#         scaler.unscale_(optimizer)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 0.999)
#         scaler.step(optimizer)
#         scaler.update()

#         iterator.set_description(
#             "epoch: {}; lr {:.7f}; Loss {:.4f} ({:.4f}); cce_loss {:.4f} ({:.4f}); Dice {:.4f} ({:.4f})".format(
#                 current_epoch,
#                 scheduler.get_lr()[-1],
#                 losses.val, losses.avg,
#                 losses1.val, losses1.avg,
#                 dices.val, dices.avg
#             )
#         )

#     scheduler.step(current_epoch)
#     print("epoch: {}; lr {:.7f}; Loss {:.4f}; CCE_loss {:.4f}; Dice {:.4f}".format(
#         current_epoch, scheduler.get_lr()[-1], losses.avg, losses1.avg, dices.avg
#     ))


# if __name__ == '__main__':
#     t0 = timeit.default_timer()
#     makedirs(models_folder, exist_ok=True)

#     seed = int(sys.argv[1])
#     cudnn.benchmark = True

#     batch_size = 1
#     val_batch_size = 1

#     print("[FDA] style images found:", len(IDABD_STYLE_FILES))
#     print("[DATA] Found pairs:", len(all_files))

#     snapshot_name = f'se154_cls_cce_{seed}_idabd_finetune'

#     # class flags for oversampling
#     file_classes = []
#     for fn in tqdm(all_files):
#         fl = np.zeros((4,), dtype=bool)
#         msk_post = cv2.imread(img_to_mask_path(pre_to_post(fn)), cv2.IMREAD_UNCHANGED)
#         if msk_post is None:
#             raise FileNotFoundError(f"Could not read: {img_to_mask_path(pre_to_post(fn))}")
#         for c in range(1, 5):
#             fl[c-1] = c in msk_post
#         file_classes.append(fl)
#     file_classes = np.asarray(file_classes)

#     train_idxs0, val_idxs = train_test_split(np.arange(len(all_files)), test_size=0.25, random_state=seed)

#     np.random.seed(seed + 123123)
#     random.seed(seed + 123123)

#     train_idxs = []
#     for i in train_idxs0:
#         train_idxs.append(i)
#         if file_classes[i, 1:].max():
#             train_idxs.append(i)
#         if file_classes[i, 1:3].max():
#             train_idxs.append(i)
#     train_idxs = np.asarray(train_idxs)

#     print('steps_per_epoch', len(train_idxs) // batch_size, 'validation_steps', len(val_idxs) // val_batch_size)

#     data_train = TrainData(train_idxs)
#     val_train = ValData(val_idxs)

#     train_data_loader = DataLoader(
#     data_train, batch_size=batch_size, num_workers=0, shuffle=True,
#     pin_memory=False, drop_last=True
#     )
#     val_data_loader = DataLoader(
#         val_train, batch_size=val_batch_size, num_workers=0, shuffle=False,
#         pin_memory=False
#     )


#     model = SeNet154_Unet_Double().cuda()

#     optimizer = AdamW(model.parameters(), lr=0.0001, weight_decay=1e-6)
#     scaler = GradScaler()

#     scheduler = lr_scheduler.MultiStepLR(
#         optimizer,
#         milestones=[3, 5, 9, 13, 17, 21, 25, 29, 33, 47, 50, 60, 70, 90, 110, 130, 150, 170, 180, 190],
#         gamma=0.5
#     )

#     # =========================
#     # PRELOAD: start from xBD stage-2 classifier weights (recommended)
#     # =========================
#     snap_to_load = f'se154_cls_cce_{seed}_tuned_best.pth'  # <-- exists in your weights/
#     ckpt_path = path.join(models_folder, snap_to_load)
#     if path.isfile(ckpt_path):
#         print("=> loading checkpoint '{}'".format(ckpt_path))
#         checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

#         loaded_dict = checkpoint['state_dict']
#         sd = model.state_dict()
#         for k in sd:
#             if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
#                 sd[k] = loaded_dict[k]
#         model.load_state_dict(sd)
#         print("loaded checkpoint '{}' (epoch {}, best_score {})".format(
#             snap_to_load, checkpoint.get('epoch', -1), checkpoint.get('best_score', -1)
#         ))
#         del loaded_dict, sd, checkpoint
#         gc.collect()
#         torch.cuda.empty_cache()
#     else:
#         print(f"[WARN] checkpoint not found: {ckpt_path} (training from scratch)")

#     # baseline eval BEFORE training (so you can compare)
#     print("\n[BASELINE on IDABD split - before fine-tune]")
#     _ = validate(model, val_data_loader)

#     seg_loss = ComboLoss({'dice': 0.5}, per_image=False).cuda()
#     ce_loss = nn.CrossEntropyLoss().cuda()

#     best_score = 0.0
#     torch.cuda.empty_cache()

#     for epoch in range(16):
#         train_epoch(epoch, seg_loss, ce_loss, model, optimizer, scheduler, train_data_loader, scaler)
#         torch.cuda.empty_cache()
#         best_score = evaluate_val(val_data_loader, best_score, model, snapshot_name, epoch)

#     elapsed = timeit.default_timer() - t0
#     print('Time: {:.3f} min'.format(elapsed / 60))

import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

from os import path, makedirs, listdir
import sys
import numpy as np
np.random.seed(1)
import random
random.seed(1)

# FDA
from glob import glob
from fda import apply_fda_uint8

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler

# ✅ Updated AMP imports (no deprecation warnings)
from torch.amp import autocast, GradScaler

from adamw import AdamW
from losses import dice_round, ComboLoss

from tqdm import tqdm
import timeit
import cv2

from zoo.models import SeNet154_Unet_Double
from imgaug import augmenters as iaa
from utils import *

# ✅ Updated skimage (square() deprecated)
from skimage.morphology import dilation, footprint_rectangle
FP5 = footprint_rectangle((5, 5))

from sklearn.model_selection import train_test_split
import gc

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# =========================
# DATA CONFIG
# =========================
train_dirs = ['idabd']
models_folder = 'weights'
input_shape = (320, 320)

# =========================
# FDA DOMAIN ADAPTATION
# =========================
USE_FDA = True
FDA_PROB = 0.5
FDA_BETA = 0.01

IDABD_STYLE_DIR = "idabd/images"
IDABD_STYLE_FILES = sorted(glob(path.join(IDABD_STYLE_DIR, "*_pre_disaster.png")))

# =========================
# PATH HELPERS
# =========================
def img_to_mask_path(fn: str) -> str:
    root = path.dirname(path.dirname(fn))
    return path.join(root, "masks", path.basename(fn))

def pre_to_post(fn: str) -> str:
    return fn.replace('_pre_disaster', '_post_disaster')

# =========================
# COLLECT FILES
# =========================
all_files = []
for d in train_dirs:
    img_dir = path.join(d, 'images')
    for f in sorted(listdir(img_dir)):
        if f.endswith('.png') and ('_pre_disaster.png' in f):
            all_files.append(path.join(img_dir, f))

if len(all_files) == 0:
    raise RuntimeError(f"No *_pre_disaster.png files found under: {train_dirs} (expected <dir>/images/*.png)")

# =========================
# DATASETS
# =========================
class TrainData(Dataset):
    def __init__(self, train_idxs):
        super().__init__()
        self.train_idxs = train_idxs
        self.elastic = iaa.ElasticTransformation(alpha=(0.25, 1.2), sigma=0.2)
        self.style_files = IDABD_STYLE_FILES

    def __len__(self):
        return len(self.train_idxs)

    def __getitem__(self, idx):
        _idx = self.train_idxs[idx]
        fn = all_files[_idx]

        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        img2 = cv2.imread(pre_to_post(fn), cv2.IMREAD_COLOR)
        if img is None or img2 is None:
            raise FileNotFoundError(f"Could not read image pair: {fn} / {pre_to_post(fn)}")

        msk0_path = img_to_mask_path(fn)
        msk1_path = img_to_mask_path(pre_to_post(fn))

        msk0 = cv2.imread(msk0_path, cv2.IMREAD_UNCHANGED)
        lbl_msk1 = cv2.imread(msk1_path, cv2.IMREAD_UNCHANGED)
        if msk0 is None or lbl_msk1 is None:
            raise FileNotFoundError(f"Could not read mask pair: {msk0_path} / {msk1_path}")

        msk1 = np.zeros_like(lbl_msk1)
        msk2 = np.zeros_like(lbl_msk1)
        msk3 = np.zeros_like(lbl_msk1)
        msk4 = np.zeros_like(lbl_msk1)
        msk1[lbl_msk1 == 1] = 255
        msk2[lbl_msk1 == 2] = 255
        msk3[lbl_msk1 == 3] = 255
        msk4[lbl_msk1 == 4] = 255

        # flips
        if random.random() > 0.5:
            img = img[::-1, ...]
            img2 = img2[::-1, ...]
            msk0 = msk0[::-1, ...]
            msk1 = msk1[::-1, ...]
            msk2 = msk2[::-1, ...]
            msk3 = msk3[::-1, ...]
            msk4 = msk4[::-1, ...]

        # rotations
        if random.random() > 0.0001:
            rot = random.randrange(4)
            if rot > 0:
                img = np.rot90(img, k=rot)
                img2 = np.rot90(img2, k=rot)
                msk0 = np.rot90(msk0, k=rot)
                msk1 = np.rot90(msk1, k=rot)
                msk2 = np.rot90(msk2, k=rot)
                msk3 = np.rot90(msk3, k=rot)
                msk4 = np.rot90(msk4, k=rot)

        # shifts
        if random.random() > 0.5:
            shift_pnt = (random.randint(-320, 320), random.randint(-320, 320))
            img = shift_image(img, shift_pnt)
            img2 = shift_image(img2, shift_pnt)
            msk0 = shift_image(msk0, shift_pnt)
            msk1 = shift_image(msk1, shift_pnt)
            msk2 = shift_image(msk2, shift_pnt)
            msk3 = shift_image(msk3, shift_pnt)
            msk4 = shift_image(msk4, shift_pnt)

        # rotate+scale
        if random.random() > 0.05:
            rot_pnt = (img.shape[0] // 2 + random.randint(-320, 320),
                       img.shape[1] // 2 + random.randint(-320, 320))
            scale = 0.9 + random.random() * 0.2
            angle = random.randint(0, 20) - 10
            if (angle != 0) or (scale != 1):
                img = rotate_image(img, angle, scale, rot_pnt)
                img2 = rotate_image(img2, angle, scale, rot_pnt)
                msk0 = rotate_image(msk0, angle, scale, rot_pnt)
                msk1 = rotate_image(msk1, angle, scale, rot_pnt)
                msk2 = rotate_image(msk2, angle, scale, rot_pnt)
                msk3 = rotate_image(msk3, angle, scale, rot_pnt)
                msk4 = rotate_image(msk4, angle, scale, rot_pnt)

        # crop
        crop_size = input_shape[0]
        if random.random() > 0.05:
            crop_size = random.randint(int(input_shape[0] / 1.15), int(input_shape[0] / 0.85))

        bst_x0 = random.randint(0, img.shape[1] - crop_size)
        bst_y0 = random.randint(0, img.shape[0] - crop_size)
        bst_sc = -1
        try_cnt = random.randint(1, 10)
        for _ in range(try_cnt):
            x0 = random.randint(0, img.shape[1] - crop_size)
            y0 = random.randint(0, img.shape[0] - crop_size)
            _sc = (msk2[y0:y0+crop_size, x0:x0+crop_size].sum() * 5 +
                   msk3[y0:y0+crop_size, x0:x0+crop_size].sum() * 5 +
                   msk4[y0:y0+crop_size, x0:x0+crop_size].sum() * 2 +
                   msk1[y0:y0+crop_size, x0:x0+crop_size].sum())
            if _sc > bst_sc:
                bst_sc = _sc
                bst_x0 = x0
                bst_y0 = y0

        x0, y0 = bst_x0, bst_y0
        img = img[y0:y0+crop_size, x0:x0+crop_size, :]
        img2 = img2[y0:y0+crop_size, x0:x0+crop_size, :]
        msk0 = msk0[y0:y0+crop_size, x0:x0+crop_size]
        msk1 = msk1[y0:y0+crop_size, x0:x0+crop_size]
        msk2 = msk2[y0:y0+crop_size, x0:x0+crop_size]
        msk3 = msk3[y0:y0+crop_size, x0:x0+crop_size]
        msk4 = msk4[y0:y0+crop_size, x0:x0+crop_size]

        if crop_size != input_shape[0]:
            img = cv2.resize(img, input_shape, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, input_shape, interpolation=cv2.INTER_LINEAR)
            msk0 = cv2.resize(msk0, input_shape, interpolation=cv2.INTER_LINEAR)
            msk1 = cv2.resize(msk1, input_shape, interpolation=cv2.INTER_LINEAR)
            msk2 = cv2.resize(msk2, input_shape, interpolation=cv2.INTER_LINEAR)
            msk3 = cv2.resize(msk3, input_shape, interpolation=cv2.INTER_LINEAR)
            msk4 = cv2.resize(msk4, input_shape, interpolation=cv2.INTER_LINEAR)

        # FDA (same style for both pre/post)
        if USE_FDA and self.style_files and (random.random() < FDA_PROB):
            tgt_fn = random.choice(self.style_files)
            tgt = cv2.imread(tgt_fn, cv2.IMREAD_COLOR)
            if tgt is not None:
                tgt = cv2.resize(tgt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
                img = apply_fda_uint8(img, tgt, beta=FDA_BETA)
                img2 = apply_fda_uint8(img2, tgt, beta=FDA_BETA)

        # color augs (same as your original)
        if random.random() > 0.9:
            img = shift_channels(img, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))
        elif random.random() > 0.9:
            img2 = shift_channels(img2, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))

        if random.random() > 0.9:
            img = change_hsv(img, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))
        elif random.random() > 0.9:
            img2 = change_hsv(img2, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))

        if random.random() > 0.9:
            if random.random() > 0.9:
                img = clahe(img)
            elif random.random() > 0.9:
                img = gauss_noise(img)
            elif random.random() > 0.9:
                img = cv2.blur(img, (3, 3))
        elif random.random() > 0.9:
            if random.random() > 0.9:
                img = saturation(img, 0.9 + random.random() * 0.2)
            elif random.random() > 0.9:
                img = brightness(img, 0.9 + random.random() * 0.2)
            elif random.random() > 0.9:
                img = contrast(img, 0.9 + random.random() * 0.2)

        if random.random() > 0.9:
            if random.random() > 0.9:
                img2 = clahe(img2)
            elif random.random() > 0.9:
                img2 = gauss_noise(img2)
            elif random.random() > 0.9:
                img2 = cv2.blur(img2, (3, 3))
        elif random.random() > 0.9:
            if random.random() > 0.9:
                img2 = saturation(img2, 0.9 + random.random() * 0.2)
            elif random.random() > 0.9:
                img2 = brightness(img2, 0.9 + random.random() * 0.2)
            elif random.random() > 0.9:
                img2 = contrast(img2, 0.9 + random.random() * 0.2)

        if random.random() > 0.9:
            el_det = self.elastic.to_deterministic()
            img = el_det.augment_image(img)

        if random.random() > 0.9:
            el_det = self.elastic.to_deterministic()
            img2 = el_det.augment_image(img2)

        # masks to channels
        msk0 = msk0[..., np.newaxis]
        msk1 = msk1[..., np.newaxis]
        msk2 = msk2[..., np.newaxis]
        msk3 = msk3[..., np.newaxis]
        msk4 = msk4[..., np.newaxis]

        msk = np.concatenate([msk0, msk1, msk2, msk3, msk4], axis=2)
        msk = (msk > 127)

        # winner-style mask logic
        msk[..., 0] = True
        msk[..., 1] = dilation(msk[..., 1], FP5)
        msk[..., 2] = dilation(msk[..., 2], FP5)
        msk[..., 3] = dilation(msk[..., 3], FP5)
        msk[..., 4] = dilation(msk[..., 4], FP5)
        msk[..., 1][msk[..., 2:].max(axis=2)] = False
        msk[..., 3][msk[..., 2]] = False
        msk[..., 4][msk[..., 2]] = False
        msk[..., 4][msk[..., 3]] = False
        msk[..., 0][msk[..., 1:].max(axis=2)] = False
        msk = msk * 1

        lbl_msk = msk.argmax(axis=2)

        img = np.concatenate([img, img2], axis=2)
        img = preprocess_inputs(img)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        msk = torch.from_numpy(msk.transpose((2, 0, 1))).long()

        return {'img': img, 'msk': msk, 'lbl_msk': lbl_msk, 'fn': fn}


class ValData(Dataset):
    def __init__(self, image_idxs):
        super().__init__()
        self.image_idxs = image_idxs

    def __len__(self):
        return len(self.image_idxs)

    def __getitem__(self, idx):
        _idx = self.image_idxs[idx]
        fn = all_files[_idx]

        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        img2 = cv2.imread(pre_to_post(fn), cv2.IMREAD_COLOR)
        if img is None or img2 is None:
            raise FileNotFoundError(f"Could not read image pair: {fn} / {pre_to_post(fn)}")

        msk0_path = img_to_mask_path(fn)
        msk1_path = img_to_mask_path(pre_to_post(fn))

        msk0 = cv2.imread(msk0_path, cv2.IMREAD_UNCHANGED)
        lbl_msk1 = cv2.imread(msk1_path, cv2.IMREAD_UNCHANGED)
        if msk0 is None or lbl_msk1 is None:
            raise FileNotFoundError(f"Could not read mask pair: {msk0_path} / {msk1_path}")

        msk1 = np.zeros_like(lbl_msk1)
        msk2 = np.zeros_like(lbl_msk1)
        msk3 = np.zeros_like(lbl_msk1)
        msk4 = np.zeros_like(lbl_msk1)
        msk1[lbl_msk1 == 1] = 255
        msk2[lbl_msk1 == 2] = 255
        msk3[lbl_msk1 == 3] = 255
        msk4[lbl_msk1 == 4] = 255

        msk0 = msk0[..., np.newaxis]
        msk1 = msk1[..., np.newaxis]
        msk2 = msk2[..., np.newaxis]
        msk3 = msk3[..., np.newaxis]
        msk4 = msk4[..., np.newaxis]

        msk = np.concatenate([msk0, msk1, msk2, msk3, msk4], axis=2)
        msk = (msk > 127)
        msk = msk * 1

        # damage GT labels 0..3, evaluated on GT building pixels only
        lbl_msk = msk[..., 1:].argmax(axis=2)

        img = np.concatenate([img, img2], axis=2)
        img = preprocess_inputs(img)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        msk = torch.from_numpy(msk.transpose((2, 0, 1))).long()

        return {'img': img, 'msk': msk, 'lbl_msk': lbl_msk, 'fn': fn}


# =========================
# VALIDATION (FIXED)
# =========================
def validate(net, data_loader, loc_thr=0.3):
    tp = np.zeros((5,), dtype=np.float64)
    fp = np.zeros((5,), dtype=np.float64)
    fn = np.zeros((5,), dtype=np.float64)

    net.eval()
    with torch.no_grad():
        for sample in tqdm(data_loader):
            msks = sample["msk"].numpy()        # (B,5,H,W) 0/1
            lbl_msk = sample["lbl_msk"].numpy() # (B,H,W) 0..3
            imgs = sample["img"].cuda(non_blocking=True)

            with autocast('cuda'):
                out = net(imgs)  # (B,5,H,W)

            # ✅ IMPORTANT: channel 0 is BACKGROUND logit in this winner-style stage-2 model
            bg_prob = torch.sigmoid(out[:, 0]).float().cpu().numpy()   # (B,H,W)
            build_prob = 1.0 - bg_prob                                 # (B,H,W)
            pr_build = (build_prob > loc_thr)                          # bool (B,H,W)

            dmg_prob = torch.softmax(out.float(), dim=1).cpu().numpy()[:, 1:, ...]  # (B,4,H,W)

            for j in range(msks.shape[0]):
                gt_build = (msks[j, 0] > 0)
                prb = pr_build[j]

                # ✅ Correct building confusion
                tp[4] += np.logical_and(gt_build, prb).sum()
                fp[4] += np.logical_and(~gt_build, prb).sum()
                fn[4] += np.logical_and(gt_build, ~prb).sum()

                # Damage evaluated only on GT building pixels
                targ = lbl_msk[j][gt_build]  # 0..3

                pred = dmg_prob[j].argmax(axis=0).astype(np.int16)  # 0..3
                # If building not predicted -> "no prediction" (-1). This avoids artificial FP.
                pred[~prb] = -1
                pred = pred[gt_build]

                for c in range(4):
                    tp[c] += np.logical_and(pred == c, targ == c).sum()
                    fp[c] += np.logical_and(pred == c, targ != c).sum()
                    fn[c] += np.logical_and(pred != c, targ == c).sum()

    d0 = 2 * tp[4] / (2 * tp[4] + fp[4] + fn[4] + 1e-9)

    f1_sc = np.zeros((4,), dtype=np.float64)
    for c in range(4):
        f1_sc[c] = 2 * tp[c] / (2 * tp[c] + fp[c] + fn[c] + 1e-9)

    f1 = 4.0 / np.sum(1.0 / (f1_sc + 1e-6))
    sc = 0.3 * d0 + 0.7 * f1

    print("Val Score: {}, Dice: {}, F1: {}, F1_0: {}, F1_1: {}, F1_2: {}, F1_3: {}".format(
        sc, d0, f1, f1_sc[0], f1_sc[1], f1_sc[2], f1_sc[3]
    ))
    return sc


def evaluate_val(data_val, best_score, model, snapshot_name, current_epoch):
    model = model.eval()
    d = validate(model, data_loader=data_val)

    if d > best_score:
        save_path = path.join(models_folder, snapshot_name + '_best.pth')
        torch.save({
            'epoch': current_epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': d,
        }, save_path)
        best_score = d
        print(f"[SAVE] {save_path}")

    print("score: {}\tscore_best: {}".format(d, best_score))
    return best_score


# =========================
# TRAIN
# =========================
def train_epoch(current_epoch, seg_loss, ce_loss, model, optimizer, scheduler, train_data_loader, scaler):
    losses = AverageMeter()
    losses1 = AverageMeter()
    dices = AverageMeter()

    iterator = tqdm(train_data_loader)
    model.train()

    for sample in iterator:
        imgs = sample["img"].cuda(non_blocking=True)
        msks = sample["msk"].cuda(non_blocking=True)
        lbl_msk = sample["lbl_msk"].cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda'):
            out = model(imgs)

            loss0 = seg_loss(out[:, 0, ...], msks[:, 0, ...])
            loss1 = seg_loss(out[:, 1, ...], msks[:, 1, ...])
            loss2 = seg_loss(out[:, 2, ...], msks[:, 2, ...])
            loss3 = seg_loss(out[:, 3, ...], msks[:, 3, ...])
            loss4 = seg_loss(out[:, 4, ...], msks[:, 4, ...])
            loss5 = ce_loss(out, lbl_msk)

            loss = 0.1 * loss0 + 0.1 * loss1 + 0.6 * loss2 + 0.3 * loss3 + 0.2 * loss4 + loss5 * 8

        with torch.no_grad():
            _probs = 1 - torch.sigmoid(out[:, 0, ...].float())
            dice_sc = 1 - dice_round(_probs, 1 - msks[:, 0, ...].float())

        losses.update(loss.item(), imgs.size(0))
        losses1.update(loss5.item(), imgs.size(0))
        dices.update(dice_sc, imgs.size(0))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.999)
        scaler.step(optimizer)
        scaler.update()

        lr_now = scheduler.get_last_lr()[-1]
        iterator.set_description(
            "epoch: {}; lr {:.7f}; Loss {:.4f} ({:.4f}); cce_loss {:.4f} ({:.4f}); Dice {:.4f} ({:.4f})".format(
                current_epoch,
                lr_now,
                losses.val, losses.avg,
                losses1.val, losses1.avg,
                dices.val, dices.avg
            )
        )

    # ✅ avoid deprecated epoch-arg usage
    scheduler.step()
    lr_now = scheduler.get_last_lr()[-1]
    print("epoch: {}; lr {:.7f}; Loss {:.4f}; CCE_loss {:.4f}; Dice {:.4f}".format(
        current_epoch, lr_now, losses.avg, losses1.avg, dices.avg
    ))


if __name__ == '__main__':
    t0 = timeit.default_timer()
    makedirs(models_folder, exist_ok=True)

    seed = int(sys.argv[1])
    cudnn.benchmark = True

    batch_size = 1
    val_batch_size = 1

    print("[FDA] style images found:", len(IDABD_STYLE_FILES))
    print("[DATA] Found pairs:", len(all_files))

    snapshot_name = f'se154_cls_cce_{seed}_idabd_finetune'

    # class flags for oversampling
    file_classes = []
    for fn in tqdm(all_files):
        fl = np.zeros((4,), dtype=bool)
        msk_post = cv2.imread(img_to_mask_path(pre_to_post(fn)), cv2.IMREAD_UNCHANGED)
        if msk_post is None:
            raise FileNotFoundError(f"Could not read: {img_to_mask_path(pre_to_post(fn))}")
        for c in range(1, 5):
            fl[c-1] = c in msk_post
        file_classes.append(fl)
    file_classes = np.asarray(file_classes)

    # ✅ keep your 0.25 split
    train_idxs0, val_idxs = train_test_split(np.arange(len(all_files)), test_size=0.25, random_state=seed)

    np.random.seed(seed + 123123)
    random.seed(seed + 123123)

    train_idxs = []
    for i in train_idxs0:
        train_idxs.append(i)
        if file_classes[i, 1:].max():
            train_idxs.append(i)
        if file_classes[i, 1:3].max():
            train_idxs.append(i)
    train_idxs = np.asarray(train_idxs)

    print('steps_per_epoch', len(train_idxs) // batch_size, 'validation_steps', len(val_idxs) // val_batch_size)

    data_train = TrainData(train_idxs)
    val_train = ValData(val_idxs)

    train_data_loader = DataLoader(
        data_train, batch_size=batch_size, num_workers=0, shuffle=True,
        pin_memory=True, drop_last=True
    )
    val_data_loader = DataLoader(
        val_train, batch_size=val_batch_size, num_workers=0, shuffle=False,
        pin_memory=True
    )

    model = SeNet154_Unet_Double().cuda()

    optimizer = AdamW(model.parameters(), lr=0.0001, weight_decay=1e-6)
    scaler = GradScaler('cuda')

    scheduler = lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[3, 5, 9, 13, 17, 21, 25, 29, 33, 47, 50, 60, 70, 90, 110, 130, 150, 170, 180, 190],
        gamma=0.5
    )

    # PRELOAD: start from xBD stage-2 classifier weights
    snap_to_load = f'se154_cls_cce_{seed}_tuned_best.pth'
    ckpt_path = path.join(models_folder, snap_to_load)
    if path.isfile(ckpt_path):
        print("=> loading checkpoint '{}'".format(ckpt_path))
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        loaded_dict = checkpoint['state_dict']
        sd = model.state_dict()
        for k in sd:
            if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
                sd[k] = loaded_dict[k]
        model.load_state_dict(sd)
        print("loaded checkpoint '{}' (epoch {}, best_score {})".format(
            snap_to_load, checkpoint.get('epoch', -1), checkpoint.get('best_score', -1)
        ))
        del loaded_dict, sd, checkpoint
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(f"[WARN] checkpoint not found: {ckpt_path} (training from scratch)")

    # baseline eval BEFORE training
    print("\n[BASELINE on IDABD split - before fine-tune]")
    _ = validate(model, val_data_loader)

    seg_loss = ComboLoss({'dice': 0.5}, per_image=False).cuda()
    ce_loss = nn.CrossEntropyLoss().cuda()

    best_score = 0.0
    torch.cuda.empty_cache()

    for epoch in range(16):
        train_epoch(epoch, seg_loss, ce_loss, model, optimizer, scheduler, train_data_loader, scaler)
        torch.cuda.empty_cache()
        best_score = evaluate_val(val_data_loader, best_score, model, snapshot_name, epoch)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
