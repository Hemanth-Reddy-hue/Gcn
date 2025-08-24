#!/usr/bin/env python3
"""
graph_classify_mutag_kfold_advanced.py

Train/validate a Graph Isomorphism Network (GIN) graph classifier on TUDataset "MUTAG"
using K-fold cross-validation with enhancements for performance and class imbalance.

- Architecture: GIN (more powerful than GCN)
- Regularization: Batch Normalization
- Readout: Jumping Knowledge-style concatenation of all layer outputs
- Imbalance Handling: Weighted Cross-Entropy Loss
- Evaluation: Macro F1-Score for model selection

Dependencies:
 - torch, torch_geometric, matplotlib
 - scikit-learn (for KFold splitting & metrics)
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# --- Hyperparameters and Setup ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DATA_ROOT = "data/TUDataset"
DATASET_NAME = "MUTAG"
BATCH_SIZE = 32
LR = 0.005  # Adjusted Learning Rate
WEIGHT_DECAY = 5e-5 # Adjusted Weight Decay
EPOCHS = 200 # Increased epochs for a more complex model
HIDDEN_DIM = 128 # Increased hidden dimensions
NUM_LAYERS = 4 # Number of GIN layers
DROPOUT = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_FOLDS = 10

# --- Model Definition (GIN with Jumping Knowledge) ---
class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Input layer
        mlp_in = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.convs.append(GINConv(mlp_in, train_eps=True))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Output layer - concatenates features from all layers
        self.lin = nn.Linear(hidden_channels * num_layers, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        h_list = [] # This is where the error was in the original context

        # 1. Obtain node embeddings from each layer
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            # Pool the representation of the current layer
            h_list.append(global_add_pool(x, batch))

        # 2. Concatenate graph embeddings from all layers (Jumping Knowledge)
        x_cat = torch.cat(h_list, dim=1)

        # 3. Apply a final classifier
        x = F.dropout(x_cat, p=self.dropout, training=self.training)
        x = self.lin(x)
        
        return x

# --- Evaluation Function ---
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss() # No weights needed for evaluation
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(out, batch.y)
            total_loss += loss.item() * batch.num_graphs
            preds = out.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total += batch.num_graphs
            all_preds.append(preds.cpu())
            all_labels.append(batch.y.cpu())

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return avg_loss, acc, all_preds, all_labels

# --- Training Function for a Single Fold ---
def train_one_fold(train_dataset, val_dataset, fold, class_weights):
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = GIN(in_channels=train_dataset.num_node_features,
                hidden_channels=HIDDEN_DIM,
                num_classes=train_dataset.num_classes,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-5)

    best_val_f1, best_state = 0.0, None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_graphs = 0.0, 0

        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            total_graphs += batch.num_graphs

        train_loss = total_loss / total_graphs
        _, _, val_preds, val_labels = evaluate(model, val_loader, DEVICE)
        val_f1 = f1_score(val_labels, val_preds, average="macro")
        
        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()
            # Optional: Add a print statement for new best model
            # print(f"*** New best model saved at epoch {epoch} with F1: {val_f1:.4f} ***")

        if epoch % 20 == 0:
            print(f"[Fold {fold}] Epoch {epoch}/{EPOCHS} | Train Loss {train_loss:.4f} | Val Macro F1 {val_f1:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model

# --- Main Execution ---
def main():
    dataset = TUDataset(root=DATA_ROOT, name=DATASET_NAME)
    print(f"Loaded dataset {DATASET_NAME}: {len(dataset)} graphs")
    print(f"Feature dim: {dataset.num_node_features}, classes: {dataset.num_classes}")

    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    fold_accuracies, fold_f1s = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n===== Starting Fold {fold+1}/{K_FOLDS} =====")
        train_dataset = dataset[train_idx.tolist()]
        val_dataset = dataset[val_idx.tolist()]

        labels = [data.y.item() for data in train_dataset]
        class_counts = np.bincount(labels, minlength=dataset.num_classes)
        # Avoid division by zero if a class is not in the training fold
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        class_weights[torch.isinf(class_weights)] = 0 
        class_weights = class_weights.to(DEVICE)
        print(f"Using class weights: {class_weights.cpu().numpy()}")

        model = train_one_fold(train_dataset, val_dataset, fold + 1, class_weights)

        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        _, val_acc, preds, labels = evaluate(model, val_loader, DEVICE)

        print(f"\n--- Fold {fold+1} Final Results ---")
        print(f"Val Acc: {val_acc:.4f}")
        print(classification_report(labels, preds, digits=4, zero_division=0))
        print("Confusion matrix:\n", confusion_matrix(labels, preds))

        fold_accuracies.append(val_acc)
        fold_f1s.append(f1_score(labels, preds, average="macro", zero_division=0))

    print("\n\n" + "="*30)
    print("    Overall K-Fold Results")
    print("="*30)
    print(f"Mean Accuracy:   {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    print(f"Mean Macro F1:   {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
    print("="*30)

if __name__ == "__main__":
    main()