# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

# --- EEG Channel to Region Mapping ---
# This mapping groups the 32 channels into 10 neuro-anatomically relevant regions.
EEG_REGIONS = {
    "Prefrontal_L": [0, 2],       # Fp1, Fp2
    "Prefrontal_R": [1, 3],       # AF3, AF4
    "Frontal_L": [4, 6, 8],       # F7, F3, FC5
    "Frontal_R": [5, 7, 9],       # F8, F4, FC6
    "Central": [10, 11, 12, 13],  # T7, C3, Cz, C4
    "Temporal_L": [14, 16, 18],   # T8, CP5, CP1
    "Temporal_R": [15, 17, 19],   # P7, P3, Pz
    "Parietal_L": [20, 22, 24],   # P4, P8, PO3
    "Parietal_R": [21, 23, 25],   # O1, Oz, O2
    "Occipital": [26, 27, 28, 29, 30, 31] # PO4, CP2, CP6, TP9, TP10, Pz
}

# --- 1. Baseline 1D-CNN for Ablation Studies A & B ---
class BaselineCNN1D(nn.Module):
    """
    A simple but robust 1D-CNN to serve as a fixed baseline for
    evaluating feature and frequency band contributions.
    """
    def __init__(self, input_features, num_classes=2):
        super(BaselineCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        flattened_size = self._get_conv_output_size(input_features)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flattened_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def _get_conv_output_size(self, input_features):
        with torch.no_grad():
            x = torch.zeros(1, 32, input_features)
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            return x.numel()

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- 2. Region-Adaptive Graph Attention Network (RG-GAT) ---
class RGGAT(nn.Module):
    """
    The proposed Region-Adaptive Graph Attention Network.
    """
    def __init__(self, initial_feature_dim, embedding_dim=64, num_classes=2):
        super(RGGAT, self).__init__()
        # --- FIX ---
        # The number of regions is now derived directly from the EEG_REGIONS dictionary
        # instead of being hardcoded, fixing the size mismatch error.
        self.num_regions = len(EEG_REGIONS)
        self.embedding_dim = embedding_dim

        self.region_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(channels) * initial_feature_dim, 128),
                nn.ReLU(),
                nn.Linear(128, embedding_dim)
            ) for region, channels in EEG_REGIONS.items()
        ])

        self.W_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_k = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.gat1 = GATConv(embedding_dim, 128, heads=8, dropout=0.6)
        self.gat2 = GATConv(128 * 8, 256, heads=1, concat=False, dropout=0.6)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        region_embeddings = []
        for i, (region_name, channels) in enumerate(EEG_REGIONS.items()):
            region_features = x[:, channels, :]
            region_features_flat = region_features.reshape(x.size(0), -1)
            encoded = self.region_encoders[i](region_features_flat)
            region_embeddings.append(encoded)

        node_features = torch.stack(region_embeddings, dim=1)

        batch_size = node_features.size(0)
        graph_representations = []
        for i in range(batch_size):
            sample_features = node_features[i]
            q = self.W_q(sample_features)
            k = self.W_k(sample_features)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.embedding_dim ** 0.5)
            adj = F.softmax(attn_scores, dim=-1)
            
            edge_index = (adj > 0).nonzero(as_tuple=False).t().contiguous()
            edge_weight = adj[edge_index[0], edge_index[1]]

            h = F.dropout(sample_features, p=0.6, training=self.training)
            h = F.elu(self.gat1(h, edge_index, edge_attr=edge_weight))
            h = F.dropout(h, p=0.6, training=self.training)
            h = self.gat2(h, edge_index, edge_attr=edge_weight)

            # The batch_idx tensor will now correctly have 10 elements, matching the 10 nodes in h.
            batch_idx = torch.zeros(self.num_regions, dtype=torch.long, device=x.device)
            pooled_graph = global_mean_pool(h, batch_idx)
            graph_representations.append(pooled_graph)

        final_representation = torch.cat(graph_representations, dim=0)
        logits = self.classifier(final_representation)
        return logits
