# main.py

import os
import pandas as pd
from data_loader import load_features_for_subject, load_labels_for_subject, preprocess_features
from train import run_cross_validation
from models import BaselineCNN1D, RGGAT

# --- Configuration ---
DATA_PATH = './features'       # Path to your extracted .npy feature files
LABEL_PATH = './labels'        # Path to your .npy label files
RESULTS_PATH = './results'     # Directory to save ablation study results
NUM_SUBJECTS = 32              # Total number of subjects in DEAP
TARGET_LABEL = 'valence'       # 'valence' or 'arousal'

def run_final_model(subject_id):
    """Train and evaluate the final RG-GAT model"""
    print(f"\n===== Running Final RG-GAT Model for Subject {subject_id} =====")
    
    # Based on your ablation results, gamma_only was the best single band.
    # We will use that for the final model evaluation.
    # You can change this to ['all'] to use all bands if you prefer.
    bands = ['gamma']
    feature_types = ['all']
    experiment_name = f'RG-GAT_final_{"_".join(bands)}'
    
    labels = load_labels_for_subject(LABEL_PATH, subject_id, TARGET_LABEL)
    features = load_features_for_subject(DATA_PATH, subject_id, bands, feature_types)
    
    if labels is None or features is None:
        print(f"Skipping subject {subject_id} due to missing data.")
        return None
        
    features = preprocess_features(features)
    
    acc, f1 = run_cross_validation(
        model_class=RGGAT,
        features=features,
        labels=labels,
        subject_id=subject_id,
        experiment_name=experiment_name,
        epochs=100, # Increased epochs for the final, more complex model
        lr=0.0005,
        patience=15 # Increased patience for the final model
    )
    
    return {'subject': subject_id, 'experiment': experiment_name, 'accuracy': acc, 'f1_score': f1}


if __name__ == '__main__':
    os.makedirs(RESULTS_PATH, exist_ok=True)

    print("="*50)
    print("Ablation study is complete. Now running the final RG-GAT model.")
    print("="*50)

    # --- Run Final RG-GAT Model for all subjects ---
    all_rggat_results = []
    # Loop through all subjects from 1 to 32
    for i in range(1, NUM_SUBJECTS + 1):
        rggat_result = run_final_model(i)
        if rggat_result is not None:
            all_rggat_results.append(rggat_result)

    if all_rggat_results:
        rggat_df = pd.DataFrame(all_rggat_results)
        rggat_df.to_csv(os.path.join(RESULTS_PATH, 'rggat_final_results.csv'), index=False)
        print("\nFull results for RG-GAT model saved to results/rggat_final_results.csv")
        
        # Print summary
        summary = rggat_df[['accuracy', 'f1_score']].mean()
        print("\n--- RG-GAT Model Summary (Average over all subjects) ---")
        print(summary)

