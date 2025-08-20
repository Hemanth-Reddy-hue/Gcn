#!/usr/bin/env python3
"""
convert_labels_va_print.py

Convert existing *_preprocessed.npz files: drop dominance & liking, create binary valence+arousal labels,
overwrite the same .npz file (optional .bak), and print dimension summaries.

Run from folder containing *_preprocessed.npz files or set PREP_DIR.
"""

import os
import numpy as np
import shutil

# ---------- User config ----------
PREP_DIR = "preprocessed/"           # "." means current working directory
FILE_SUFFIX = "_preprocessed.npz"
BACKUP_ORIGINAL = True   # if True, create filename + ".bak" before overwriting
NEUTRAL_POLICY = "keep"  # "keep" (treat 5 as low) or "drop" (remove windows with valence==5 or arousal==5)
THRESH = 5.0             # threshold for binary: >THRESH => 1, else 0
VERBOSE = True
# ----------------------------------

def ensure_2d_labels(orig_labels):
    """Convert object/1D label arrays to (n,4) numeric 2D array if needed."""
    lab = np.asarray(orig_labels)
    if lab.ndim == 2:
        return lab
    if lab.ndim == 1 and lab.dtype == object:
        # object array of per-window arrays -> vstack
        return np.vstack(lab)
    raise ValueError("Unsupported label array shape/dtype: " + str(lab.shape) + " " + str(lab.dtype))

def expand_trial_labels_if_needed(orig_labels, n_windows, per_trial_counts=None):
    """
    If orig_labels is per-trial (40,4), expand to per-window using per_trial_counts if available,
    otherwise assume equal replication.
    """
    orig = np.asarray(orig_labels)
    if orig.shape[0] == n_windows:
        return orig
    if orig.shape[0] == 40:
        if per_trial_counts is None:
            per = n_windows // 40
            counts = [per] * 40
            rem = n_windows - per * 40
            for i in range(rem):
                counts[i] += 1
        else:
            counts = list(per_trial_counts)
            # if provided counts don't sum to n_windows, attempt to adjust evenly
            if sum(counts) != n_windows:
                per = n_windows // 40
                counts = [per] * 40
                rem = n_windows - per * 40
                for i in range(rem):
                    counts[i] += 1
        expanded = []
        for t in range(40):
            expanded.extend([orig[t]] * counts[t])
        expanded = np.array(expanded)
        if expanded.shape[0] != n_windows:
            expanded = expanded[:n_windows]
        return expanded
    else:
        raise ValueError(f"Label length ({orig.shape[0]}) does not match n_windows ({n_windows}) and is not per-trial (40).")

def process_file(path: str, backup: bool = True, neutral_policy: str = "keep", thresh: float = 5.0):
    fname = os.path.basename(path)
    if backup:
        bak_path = path + ".bak"
        if not os.path.exists(bak_path):
            shutil.copy2(path, bak_path)
            if VERBOSE:
                print(f"[backup] {fname} -> {os.path.basename(bak_path)}")
        else:
            if VERBOSE:
                print(f"[backup] backup already exists: {os.path.basename(bak_path)}")

    # load file
    d = np.load(path, allow_pickle=True)
    if VERBOSE:
        print(f"\nProcessing file: {fname}")
        print("  Keys in file:", d.files)

    if 'data' not in d:
        raise KeyError(f"'data' not found in {fname}")

    X = d['data']  # (n_windows, n_channels, n_samples)
    n_windows = X.shape[0]
    orig_labels = None
    if 'labels' in d:
        orig_labels = d['labels']
    elif 'labels_orig' in d:
        orig_labels = d['labels_orig']
    elif 'labels_full' in d:
        orig_labels = d['labels_full']
    else:
        raise KeyError(f"No 'labels' or 'labels_orig' found in {fname}")

    # preserve any metadata we want to carry forward
    per_trial_counts = d.get('per_trial_epoch_counts', None)
    trial_map = d.get('trial_map', None)
    sfreq = d.get('sfreq', None)
    other_keys = {k: d[k] for k in d.files if k not in ('data', 'labels', 'labels_orig', 'labels_full')}

    # normalize orig_labels to 2D numeric (n,4)
    orig_labels_2d = ensure_2d_labels(orig_labels)
    # if per-trial -> expand
    if orig_labels_2d.shape[0] != n_windows:
        orig_labels_2d = expand_trial_labels_if_needed(orig_labels_2d, n_windows, per_trial_counts)

    if VERBOSE:
        print(f"  Original shapes -> data: {X.shape}, labels (original): {orig_labels_2d.shape}")

    # Extract valence & arousal and do binary mapping
    val = orig_labels_2d[:, 0].astype(float)
    aro = orig_labels_2d[:, 1].astype(float)

    if neutral_policy == "drop":
        keep_mask = np.logical_and(val != thresh, aro != thresh)
    else:
        keep_mask = np.ones(len(val), dtype=bool)

    val_bin = (val > thresh).astype(np.int64)
    aro_bin = (aro > thresh).astype(np.int64)

    val_bin_out = val_bin[keep_mask]
    aro_bin_out = aro_bin[keep_mask]
    labels_va = np.stack([val_bin_out, aro_bin_out], axis=1)  # (n_kept, 2)

    # Prepare arrays to save (overwrite same file)
    save_dict = {}
    # apply keep_mask to data and other window-level arrays if we dropped anything
    if np.all(keep_mask):
        save_dict['data'] = X
        save_dict['labels'] = labels_va
        # keep original labels for traceability under a different key
        save_dict['labels_orig'] = orig_labels_2d
        if trial_map is not None:
            save_dict['trial_map'] = trial_map
        if per_trial_counts is not None:
            save_dict['per_trial_epoch_counts'] = per_trial_counts
        if sfreq is not None:
            save_dict['sfreq'] = sfreq
        # add other keys unchanged
        for k, v in other_keys.items():
            save_dict[k] = v
    else:
        save_dict['data'] = X[keep_mask]
        save_dict['labels'] = labels_va
        save_dict['labels_orig'] = orig_labels_2d[keep_mask]
        if trial_map is not None:
            save_dict['trial_map'] = trial_map[keep_mask]
        # per_trial_epoch_counts stays as original (optional to recompute)
        if per_trial_counts is not None:
            save_dict['per_trial_epoch_counts'] = per_trial_counts
        if sfreq is not None:
            save_dict['sfreq'] = sfreq
        for k, v in other_keys.items():
            # if key is window-aligned, try to mask it; otherwise keep as-is
            if isinstance(v, np.ndarray) and v.shape[0] == n_windows:
                save_dict[k] = v[keep_mask]
            else:
                save_dict[k] = v

    # overwrite same file
    np.savez_compressed(path, **save_dict)

    # print final summary
    print(f"[saved] {fname} -> data: {save_dict['data'].shape}, labels (valence+arousal): {save_dict['labels'].shape}")
    val_counts = np.bincount(save_dict['labels'][:, 0]) if save_dict['labels'].shape[0] > 0 else np.array([0, 0])
    aro_counts = np.bincount(save_dict['labels'][:, 1]) if save_dict['labels'].shape[0] > 0 else np.array([0, 0])
    print(f"  Valence counts (0/1): {val_counts.tolist()}, Arousal counts (0/1): {aro_counts.tolist()}")
    return path

def main():
    files = sorted([f for f in os.listdir(PREP_DIR) if f.endswith(FILE_SUFFIX)])
    if not files:
        raise FileNotFoundError(f"No files ending with '{FILE_SUFFIX}' found in {PREP_DIR}")
    for fname in files:
        full = os.path.join(PREP_DIR, fname)
        try:
            process_file(full, backup=BACKUP_ORIGINAL, neutral_policy=NEUTRAL_POLICY, thresh=THRESH)
        except Exception as e:
            print(f"[ERROR] processing {fname}: {e}")

main()
