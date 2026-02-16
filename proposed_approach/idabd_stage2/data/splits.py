from idabd_stage2.data.dataset_stage2_damage import has_destroyed

def ensure_destroyed_in_splits(train_tr, val_tr, min_train: int, min_val: int):
    """
    Mutates train_tr/val_tr in-place to guarantee at least:
      - min_val destroyed tiles in VAL
      - min_train destroyed tiles in TRAIN
    Uses simple swapping to avoid changing split sizes.
    """
    def destroyed(tr):
        return [t for t in tr if has_destroyed(t[2])]

    # Ensure VAL
    if min_val > 0:
        val_d = destroyed(val_tr)
        if len(val_d) < min_val:
            train_d = destroyed(train_tr)
            need = min_val - len(val_d)
            moved = 0
            for t in train_d:
                if moved >= need or len(val_tr) == 0:
                    break
                val_swap = val_tr[0]
                val_tr.remove(val_swap)
                train_tr.append(val_swap)

                if t in train_tr:
                    train_tr.remove(t)
                    val_tr.append(t)
                    moved += 1
            print(f"[SPLIT] ensured VAL destroyed tiles: {len(destroyed(val_tr))} (moved={moved})")

    # Ensure TRAIN
    if min_train > 0:
        train_d = destroyed(train_tr)
        if len(train_d) < min_train:
            val_d = destroyed(val_tr)
            need = min_train - len(train_d)
            moved = 0
            for t in val_d:
                if moved >= need or len(train_tr) == 0:
                    break
                train_swap = train_tr[0]
                train_tr.remove(train_swap)
                val_tr.append(train_swap)

                if t in val_tr:
                    val_tr.remove(t)
                    train_tr.append(t)
                    moved += 1
            print(f"[SPLIT] ensured TRAIN destroyed tiles: {len(destroyed(train_tr))} (moved={moved})")
