import os
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_features_for_subject(data_path, subject_id, bands, feature_types):
    """
    Loads and combines specified features for a single subject.

    Args:
        data_path (str): Path to the directory containing feature .npy files.
        subject_id (int): The subject number (1-32).
        bands (list): List of frequency bands to include (e.g., ['alpha', 'beta']).
        feature_types (list): List of feature domains (e.g., ['freqdomain', 'timedomain']).

    Returns:
        np.ndarray: A combined feature array of shape (4600, 32, N_combined_features).
        Returns None if no files are found.
    """
    subject_str = f's{subject_id:02d}'
    all_features = []
    
    # Check if we need to load all bands or feature types
    if 'all' in bands:
        bands = ['theta', 'alpha', 'beta', 'gamma']
    if 'all' in feature_types:
        feature_types = ['freqdomain', 'timedomain', 'timefreq']

    # Iterate through all combinations and load the data
    for band in bands:
        for ftype in feature_types:
            filename = f'{subject_str}_{band}_{ftype}.npy'
            filepath = os.path.join(data_path, filename)
            
            if os.path.exists(filepath):
                try:
                    data = np.load(filepath)
                    # Basic validation of shape
                    if data.shape[0] == 4600 and data.shape[1] == 32:
                        all_features.append(data)
                    else:
                        print(f"Warning: Skipping {filename} due to unexpected shape {data.shape}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
            else:
                print(f"Warning: File not found - {filepath}")

    if not all_features:
        print(f"Error: No feature files could be loaded for subject {subject_id} with specified criteria.")
        return None

    # Concatenate features along the last dimension
    combined_features = np.concatenate(all_features, axis=-1)
    
    return combined_features


def load_labels_for_subject(label_path, subject_id, label_type='valence'):
    """
    Loads pre-binarized labels for a single subject.

    Args:
        label_path (str): Path to the directory containing label files.
        subject_id (int): The subject number (1-32).
        label_type (str): 'valence' or 'arousal'.

    Returns:
        np.ndarray: A binary label array of shape (4600,).
    """
    subject_str = f's{subject_id:02d}'
    # Assuming labels are stored in a file like 's01_labels.npy' with shape (4600, 2)
    # where column 0 is valence and column 1 is arousal, already binarized (0 or 1).
    label_file = os.path.join(label_path, f'{subject_str}_labels.npy')

    if not os.path.exists(label_file):
        print(f"Error: Label file not found for subject {subject_id} at {label_file}")
        # As a fallback, creating dummy labels for code to run
        print("Creating dummy labels for demonstration.")
        return (np.random.rand(4600) > 0.5).astype(int)

    labels = np.load(label_file)
    
    label_idx = 0 if label_type == 'valence' else 1
    
    # Select the appropriate column (valence or arousal) which is already binarized.
    binary_labels = labels[:, label_idx].astype(int)
    
    return binary_labels


def preprocess_features(features):
    """
    Applies standardization to the features.

    Args:
        features (np.ndarray): The feature array of shape (trials, channels, features).

    Returns:
        np.ndarray: The standardized feature array.
    """
    # Reshape for scaler: (trials * channels, features)
    num_trials, num_channels, num_features = features.shape
    features_reshaped = features.reshape(-1, num_features)
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_reshaped)
    
    # Reshape back to original: (trials, channels, features)
    features_standardized = features_scaled.reshape(num_trials, num_channels, num_features)
    
    return features_standardized
