#!/usr/bin/env python3
"""
extract_features_resume.py

Compute per-band features (frequency, time-domain, time-frequency) for all subjects
with preprocessed data saved as *_preprocessed.npz in PREP_DIR.

This script RESUMES work: it will skip band outputs that already exist in SAVE_DIR
and only compute missing band files for each subject.
"""

import os
import numpy as np
import pywt
from scipy.signal import welch
from scipy.stats import skew, kurtosis

# -----------------------
# User config
# -----------------------
PREP_DIR = "preprocessed"   # directory containing *_preprocessed.npz files
SAVE_DIR = "features"       # where to save feature .npy files
os.makedirs(SAVE_DIR, exist_ok=True)

FS = 128  # sampling frequency after preprocessing

BANDS = {
    "theta": (4, 7),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 50)
}

WAVELET = "db4"
WAVELET_LEVEL = 4  # level used in pywt.wavedec -> n_subbands = level + 1

VERBOSE = True
# -----------------------


# ---------- feature functions ----------
def bandpower(epoch, fs, band):
    """
    epoch: (n_channels, n_samples)
    returns: (n_channels,) absolute band power (integral of PSD over band)
    """
    f, Pxx = welch(epoch, fs=fs, nperseg=min(epoch.shape[1], fs*2), axis=-1)
    freq_ix = np.logical_and(f >= band[0], f <= band[1])
    bandp = np.trapz(Pxx[:, freq_ix], f[freq_ix], axis=1)
    return bandp  # shape (n_channels,)


def differential_entropy(epoch, fs, band):
    bp = bandpower(epoch, fs, band)
    return np.log(bp + 1e-12)  # small eps to avoid -inf


def time_features(epoch):
    """
    epoch: (n_channels, n_samples)
    returns: (n_channels, 7) -> [mean, var, skew, kurtosis, activity, mobility, complexity]
    """
    feats = []
    for ch in epoch:
        mean_val = np.mean(ch)
        var_val = np.var(ch)
        skew_val = skew(ch)
        kurt_val = kurtosis(ch)

        diff1 = np.diff(ch)
        diff2 = np.diff(diff1) if diff1.size > 1 else np.zeros(1)
        activity = var_val
        mobility = np.sqrt(np.var(diff1) / activity) if activity != 0 else 0.0
        complexity = (np.sqrt(np.var(diff2) / np.var(diff1)) / mobility) if (np.var(diff1) != 0 and mobility != 0) else 0.0

        feats.append([mean_val, var_val, skew_val, kurt_val, activity, mobility, complexity])
    return np.asarray(feats)  # (n_channels, 7)


def time_frequency_features(epoch, wavelet=WAVELET, level=WAVELET_LEVEL):
    """
    epoch: (n_channels, n_samples)
    returns: (n_channels, n_subbands) energies of wavelet coeffs (level+1 subbands)
    """
    feats = []
    for ch in epoch:
        coeffs = pywt.wavedec(ch, wavelet, level=level)
        energies = [float(np.sum(np.square(c))) for c in coeffs]  # scalars
        feats.append(energies)
    return np.asarray(feats)  # (n_channels, n_subbands)


# ---------- helper to check existing files ----------
def band_output_files_exist(subj_name, band_name, out_dir):
    """Return True if all 3 expected files exist for subj+band in out_dir."""
    fn_freq = os.path.join(out_dir, f"{subj_name}_{band_name}_freqdomain.npy")
    fn_time = os.path.join(out_dir, f"{subj_name}_{band_name}_timedomain.npy")
    fn_tf   = os.path.join(out_dir, f"{subj_name}_{band_name}_timefreq.npy")
    return os.path.exists(fn_freq) and os.path.exists(fn_time) and os.path.exists(fn_tf)


# ---------- per-subject extraction (single band or all bands) ----------
def extract_features_for_subject(npz_path, save_dir=SAVE_DIR, bands_to_process=None):
    subj_base = os.path.basename(npz_path)
    # derive subj_name like "s01" or "s26" from "s01_preprocessed.npz"
    subj_name = subj_base.split(".")[0].replace("_preprocessed", "")
    data = np.load(npz_path, allow_pickle=True)
    if "data" not in data:
        raise KeyError(f"'data' key not found in {npz_path}")
    X = data["data"]    # (n_epochs, n_channels, n_samples)
    labels = data.get("labels", None)  # (n_epochs, 2) maybe present

    n_epochs, n_ch, n_samples = X.shape
    if VERBOSE:
        print(f"\nSubject {subj_name}: X={X.shape}, labels={None if labels is None else labels.shape}")

    # pick bands to process
    bands_iter = bands_to_process if bands_to_process is not None else list(BANDS.keys())

    processed_bands = []
    for band_name in bands_iter:
        if band_name not in BANDS:
            print(f"[WARN] Unknown band '{band_name}', skipping.")
            continue

        # skip if outputs already exist
        if band_output_files_exist(subj_name, band_name, save_dir):
            if VERBOSE:
                print(f"  Skipping {subj_name} {band_name}: output files already exist.")
            continue

        band_range = BANDS[band_name]
        freq_list = []
        time_list = []
        tf_list = []

        # iterate epochs
        for idx in range(n_epochs):
            epoch = X[idx]  # (n_channels, n_samples)

            # freq features
            bp = bandpower(epoch, FS, band_range)               # (32,)
            de = differential_entropy(epoch, FS, band_range)    # (32,)
            freq_node_feats = np.stack([bp, de], axis=-1)       # (32, 2)
            freq_list.append(freq_node_feats)

            # time features
            t_feats = time_features(epoch)                      # (32, 7)
            time_list.append(t_feats)

            # time-frequency (wavelet)
            tf_feats = time_frequency_features(epoch)           # (32, n_subbands)
            tf_list.append(tf_feats)

        # convert lists -> arrays
        freq_arr = np.asarray(freq_list)   # (n_epochs, n_ch, 2)
        time_arr = np.asarray(time_list)   # (n_epochs, n_ch, 7)
        tf_arr = np.asarray(tf_list)       # (n_epochs, n_ch, n_subbands)

        # Save arrays for this subject+band
        fname_freq = os.path.join(save_dir, f"{subj_name}_{band_name}_freqdomain.npy")
        fname_time = os.path.join(save_dir, f"{subj_name}_{band_name}_timedomain.npy")
        fname_tf   = os.path.join(save_dir, f"{subj_name}_{band_name}_timefreq.npy")

        np.save(fname_freq, freq_arr)
        np.save(fname_time, time_arr)
        np.save(fname_tf, tf_arr)

        print(f"  Saved {subj_name} {band_name}: freq={freq_arr.shape}, time={time_arr.shape}, tf={tf_arr.shape}")
        processed_bands.append(band_name)

    return processed_bands


# ---------------- main: loop all subjects ----------------
if __name__ == "__main__":
    # Read subject files
    files = sorted([f for f in os.listdir(PREP_DIR) if f.endswith("_preprocessed.npz")])
    if len(files) == 0:
        files = sorted([f for f in os.listdir(PREP_DIR) if f.endswith(".npz")])
    if len(files) == 0:
        raise FileNotFoundError(f"No .npz files found in {PREP_DIR}")

    print(f"Found {len(files)} preprocessed subject files in {PREP_DIR}.")

    # Optional: resume from specific subject number (None => process all)
    # Example: set start_subj = 26 to skip s01..s25 entirely
    start_subj = 26   # set to e.g. 26 to skip earlier subjects
    # Optional: restrict to certain bands only (None => all bands)
    bands_to_run = None  # e.g. ["theta", "alpha"] or None

    # Iterate files
    summary = {}
    for fname in files:
        # derive subject numeric id if filename contains "sNN"
        # robust parsing: find token starting with 's' followed by 1-3 digits
        subj_id = None
        base = fname.split(".")[0]
        tokens = base.split("_")
        for tok in tokens:
            if tok.startswith("s") and tok[1:].isdigit():
                try:
                    subj_id = int(tok[1:])
                    break
                except Exception:
                    pass

        if start_subj is not None and subj_id is not None and subj_id < start_subj:
            # skip earlier subjects
            if VERBOSE:
                print(f"Skipping {fname} (subject {subj_id} < start_subj={start_subj})")
            continue

        path = os.path.join(PREP_DIR, fname)
        try:
            processed = extract_features_for_subject(path, save_dir=SAVE_DIR, bands_to_process=bands_to_run)
            summary[fname] = processed
        except Exception as e:
            print(f"[ERROR] {fname}: {e}")

    # Print summary
    print("\n=== Summary of processed bands per file ===")
    for k, v in summary.items():
        print(f"{k}: processed bands -> {v if v else 'skipped or already present'}")

    print("\nAll done. Features saved to:", os.path.abspath(SAVE_DIR))
