from __future__ import annotations
import random
from proposed_approach.data.idabd_preprocess import has_destroyed

def split_triplets(triplets, val_ratio: float, test_ratio: float, seed: int):
    rng = random.Random(seed)
    items = list(triplets)
    rng.shuffle(items)
    n = len(items)

    n_test = max(1, int(n * test_ratio))
    n_val  = max(1, int(n * val_ratio))
    if n_test + n_val >= n:
        n_test = max(1, min(n_test, n - 2))
        n_val  = max(1, min(n_val,  n - 1 - n_test))

    test_tr  = items[:n_test]
    val_tr   = items[n_test:n_test+n_val]
    train_tr = items[n_test+n_val:]
    return train_tr, val_tr, test_tr

def ensure_min_destroyed_in_split(train_tr, val_tr, min_train: int, min_val: int):
    """
    Swap tiles to ensure minimum destroyed presence in train/val.
    """
    # Ensure VAL
    if min_val > 0:
        val_d = [t for t in val_tr if has_destroyed(t[2])]
        if len(val_d) < min_val:
            train_d = [t for t in train_tr if has_destroyed(t[2])]
            need = min_val - len(val_d)
            moved = 0
            for t in train_d:
                if moved >= need or len(val_tr) == 0:
                    break
                # swap with first val
                val_swap = val_tr[0]
                val_tr.remove(val_swap)
                train_tr.append(val_swap)

                train_tr.remove(t)
                val_tr.append(t)
                moved += 1

    # Ensure TRAIN
    if min_train > 0:
        train_d = [t for t in train_tr if has_destroyed(t[2])]
        if len(train_d) < min_train:
            val_d = [t for t in val_tr if has_destroyed(t[2])]
            need = min_train - len(train_d)
            moved = 0
            for t in val_d:
                if moved >= need or len(train_tr) == 0:
                    break
                train_swap = train_tr[0]
                train_tr.remove(train_swap)
                val_tr.append(train_swap)

                val_tr.remove(t)
                train_tr.append(t)
                moved += 1

    return train_tr, val_tr