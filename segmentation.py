#!/usr/bin/env python3
"""
preprocess_all_subjects_with_ica.py

Full preprocessing for all DEAP subjects:
 - baseline removal (first 3s)
 - optional resample to TARGET_SF (DEAP .dat is already 128 Hz)
 - bandpass 0.5-50 Hz + notch 50/60
 - fit ICA per subject on concatenated trials and (optionally) auto-remove EOG ICs
 - segment each 60s trial into 3s windows with 2.5s overlap (step=0.5s)
 - optional high-variance epoch rejection (conservative)
 - save per-subject .npz files and print shapes
"""

import os
import pickle
import numpy as np
from scipy.signal import resample
import mne
from typing import Tuple

# --------------------------
# CONFIG
# --------------------------
DATA_DIR = "EEG_dataset"   # folder containing s01.dat, s02.dat, ...
SAVE_DIR = "preprocessed"
os.makedirs(SAVE_DIR, exist_ok=True)

# DEAP specifics
SAMPLING_ORIG = 128   # DEAP .dat sampling rate
TARGET_SF = 128       # desired sampling rate (no resample if equal)
EEG_CHANNELS = 32
BASELINE_SEC = 3      # remove first 3 seconds (baseline)
TRIAL_SEC = 60        # length after baseline removal (60s)

# segmentation params (3s windows, 2.5s overlap => step=0.5s)
WINDOW_SEC = 3.0
STEP_SEC = 0.5

# ICA / artifact removal options
AUTO_ICA_REMOVE = True      # Fit ICA and try to auto-remove ocular components (conservative)
ICA_N_COMPONENTS = 20       # number of ICA components to fit (<= n_channels)
ICA_CORR_THRESH = 0.6       # abs(corr) threshold vs frontal proxy to mark IC as ocular
FRONTAL_CHS = ('Fp1', 'Fp2')  # DEAP standard channel names (we'll use these when available)
USE_ICLABEL = False         # set True if you have ICLabel plugin installed (optional)

# Epoch rejection (conservative)
REJECT_HIGH_VAR = False
REJECT_Z_THRESH = 4.0      # only reject epoch if z-score of its variance across epochs > this

# Debugging verbosity
PRINT_TRIAL_DEBUG = False

# Standard DEAP channel names (in order). We'll use these when creating MNE info.
CHAN_NAMES = [
 'Fp1','AF3','F3','F7','FC5','FC1','C3','T7','CP5','CP1','P3','P7','PO3','O1','Oz','Pz',
 'Fp2','AF4','Fz','F4','F8','FC6','FC2','Cz','C4','T8','CP6','CP2','P4','P8','PO4','O2'
]

# --------------------------
# helpers
# --------------------------
def segment_epochs_from_array(data: np.ndarray, window_sec: float, step_sec: float, fs: int) -> Tuple[np.ndarray,int,int,int]:
    """
    Segment (n_channels, n_samples) into overlapping windows.
    Returns (epochs_array, n_epochs, win_samp, step_samp)
    """
    n_channels, n_samples = data.shape
    win_samp = int(round(window_sec * fs))
    step_samp = int(round(step_sec * fs))
    if win_samp <= 0 or step_samp <= 0:
        raise ValueError("window_sec and step_sec must be positive")
    starts = list(range(0, n_samples - win_samp + 1, step_samp))
    segments = [data[:, s:s + win_samp] for s in starts]
    return np.array(segments), len(starts), win_samp, step_samp

def auto_ica_exclude(raw: mne.io.Raw, picks=None, n_components=20, corr_thresh=ICA_CORR_THRESH, frontal_chs=FRONTAL_CHS, use_iclabel=USE_ICLABEL) -> Tuple[mne.io.Raw, list]:
    """
    Fit ICA on raw and try to exclude ocular components.
    Returns (raw_clean, excluded_list)
    - If ICLabel is available and use_iclabel=True, tries that.
    - Otherwise uses correlation of IC timecourses with frontal proxy (Fp1/Fp2).
    """
    picks_eeg = picks if picks is not None else mne.pick_types(raw.info, eeg=True)
    n_comp_fit = min(n_components, len(picks_eeg))
    ica = mne.preprocessing.ICA(n_components=n_comp_fit, random_state=97, max_iter="auto", verbose=False)
    ica.fit(raw, picks=picks_eeg)

    excluded = set()

    # Try ICLabel if desired and available
    if use_iclabel:
        try:
            from mne_icalabel import label_components  # requires mne-iclabel plugin
            labels = label_components(ica, raw)  # plugin-specific API
            # labels expected as list of dicts/arrays; we conservatively remove ICs with 'eye' prob > 0.8
            for idx, probs in enumerate(labels):
                if isinstance(probs, dict):
                    if probs.get('eye', 0) >= 0.8 or probs.get('muscle', 0) >= 0.8:
                        excluded.add(idx)
                else:
                    # unknown format -> skip
                    pass
        except Exception:
            # fallback to correlation method
            use_iclabel = False

    # Fallback: correlation with frontal proxy (Fp1/Fp2)
    if not use_iclabel:
        # check frontal channels in raw
        available_frontal = [ch for ch in frontal_chs if ch in raw.ch_names]
        if available_frontal:
            eog_proxy = np.mean(raw.get_data(picks=[raw.ch_names.index(ch) for ch in available_frontal]), axis=0)
            sources = ica.get_sources(raw).get_data()  # shape (n_components, n_times)
            for idx in range(sources.shape[0]):
                comp_ts = sources[idx]
                # protect against constant series
                if np.std(comp_ts) < 1e-12 or np.std(eog_proxy) < 1e-12:
                    continue
                r = np.corrcoef(comp_ts, eog_proxy)[0,1]
                if np.abs(r) >= corr_thresh:
                    excluded.add(idx)
        else:
            # No frontal channels; cannot auto-detect reliably
            pass

    excluded = sorted(list(excluded))
    if excluded:
        ica.exclude = excluded
        try:
            raw_clean = ica.apply(raw.copy())
            return raw_clean, excluded
        except Exception as e:
            print("[ICA] apply failed:", e)
            return raw, []
    else:
        return raw, []

# --------------------------
# per-subject preprocessing
# --------------------------
def preprocess_one_subject(dat_path: str, save_dir: str):
    with open(dat_path, "rb") as f:
        subj = pickle.load(f, encoding="latin1")
    raw_data = subj["data"]   # shape (40, 40, 8064)
    labels   = subj["labels"] # shape (40, 4)

    n_trials = raw_data.shape[0]
    print(f"\nProcessing {os.path.basename(dat_path)}  - trials: {n_trials}")

    # Keep only EEG channels (first 32 in DEAP)
    trials_eeg = raw_data[:, :EEG_CHANNELS, :]  # shape (40, 32, 8064)

    # Remove baseline (first 3s = 3 * SAMPLING_ORIG samples) and optionally resample each trial
    baseline_samples = int(round(BASELINE_SEC * SAMPLING_ORIG))
    per_trial_samples_after = int(round(TRIAL_SEC * SAMPLING_ORIG))  # expected 60*128 = 7680
    trials_post = []
    for t in range(n_trials):
        trial = trials_eeg[t]
        trial_after = trial[:, baseline_samples:]
        if trial_after.shape[1] != per_trial_samples_after:
            print(f"[WARN] trial {t}: expected {per_trial_samples_after} samples after baseline but got {trial_after.shape[1]}")
        # resample if needed
        if SAMPLING_ORIG != TARGET_SF:
            new_len = int(round(trial_after.shape[1] * (TARGET_SF / SAMPLING_ORIG)))
            trial_rs = resample(trial_after, new_len, axis=1)
            trials_post.append(trial_rs)
        else:
            trials_post.append(trial_after.copy())

    # Concatenate trials along time to form one long continuous signal for ICA fitting
    concatenated = np.concatenate(trials_post, axis=1)  # shape (32, n_trials * per_trial_samples_after)
    fs = TARGET_SF if SAMPLING_ORIG != TARGET_SF else SAMPLING_ORIG

    # Create MNE RawArray for filtering & ICA
    info = mne.create_info(ch_names=CHAN_NAMES[:EEG_CHANNELS], sfreq=fs, ch_types='eeg')
    raw_concat = mne.io.RawArray(concatenated, info, verbose=False)

    # Filter / notch before ICA (recommended)
    raw_concat.filter(l_freq=0.5, h_freq=50.0, fir_design="firwin", verbose=False)
    raw_concat.notch_filter(freqs=[50.0, 60.0], verbose=False)

    # Fit ICA once per subject and optionally auto-remove components
    excluded_components = []
    if AUTO_ICA_REMOVE:
        try:
            raw_concat_clean, excluded_components = auto_ica_exclude(raw_concat,
                                                                     picks=None,
                                                                     n_components=ICA_N_COMPONENTS,
                                                                     corr_thresh=ICA_CORR_THRESH,
                                                                     frontal_chs=FRONTAL_CHS,
                                                                     use_iclabel=USE_ICLABEL)
            print(f"ICA auto-excluded components: {excluded_components}")
        except Exception as e:
            print("[ICA] auto exclusion failed:", e)
            raw_concat_clean = raw_concat
            excluded_components = []
    else:
        raw_concat_clean = raw_concat

    # After cleaning, split concatenated back into per-trial arrays
    trial_len_samples = int(round(TRIAL_SEC * fs))
    trials_clean = []
    for t in range(n_trials):
        start = t * trial_len_samples
        stop  = start + trial_len_samples
        trials_clean.append(raw_concat_clean.get_data()[:, start:stop])

    # Now segment each trial into windows and optionally reject high-variance epochs
    all_epochs = []
    all_epoch_labels = []
    per_trial_epoch_counts = []

    for t in range(n_trials):
        trial_arr = trials_clean[t]  # (n_ch, n_samples)
        epochs, n_ep, win_samp, step_samp = segment_epochs_from_array(trial_arr, WINDOW_SEC, STEP_SEC, fs)
        # optional epoch rejection based on variance z-score (conservative)
        if REJECT_HIGH_VAR:
            # compute per-epoch variance across channels and samples
            var_per_epoch = np.var(epochs.reshape(epochs.shape[0], -1), axis=1)
            z = (var_per_epoch - np.mean(var_per_epoch)) / (np.std(var_per_epoch) + 1e-12)
            keep_mask = np.abs(z) < REJECT_Z_THRESH
            epochs = epochs[keep_mask]
            n_ep = epochs.shape[0]

        if PRINT_TRIAL_DEBUG:
            print(f"Trial {t:02d}: samples={trial_arr.shape[1]}, win={win_samp}, step={step_samp}, epochs={n_ep}")

        all_epochs.append(epochs)
        all_epoch_labels.extend([labels[t]] * n_ep)
        per_trial_epoch_counts.append(n_ep)

    all_epochs = np.vstack(all_epochs)   # (total_epochs, n_ch, win_samp)
    all_epoch_labels = np.array(all_epoch_labels)

    # Save
    subj_name = os.path.splitext(os.path.basename(dat_path))[0]
    save_path = os.path.join(save_dir, f"{subj_name}_preprocessed.npz")
    np.savez_compressed(save_path, data=all_epochs, labels=all_epoch_labels, sfreq=fs, excluded_ica=excluded_components, per_trial_epoch_counts=per_trial_epoch_counts)

    print(f"Saved {subj_name}: data {all_epochs.shape}, labels {all_epoch_labels.shape}, per-trial-epochs {per_trial_epoch_counts}")
    return all_epochs, all_epoch_labels, per_trial_epoch_counts

# --------------------------
# run all subjects
# --------------------------
if __name__ == "__main__":
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".dat")])
    summary = []
    for f in files:
        dat_path = os.path.join(DATA_DIR, f)
        epochs, labels, per_trial_counts = preprocess_one_subject(dat_path, SAVE_DIR)
        summary.append((f, epochs.shape, labels.shape, per_trial_counts))

    # Print concise summary table
    print("\n=== Summary (subject -> data shape, labels shape, per-trial epoch counts) ===")
    for subj, shape_data, shape_labels, counts in summary:
        print(f"{subj}: data {shape_data}, labels {shape_labels}, epochs_per_trial(first/last) {counts[0]}/{counts[-1]}")
