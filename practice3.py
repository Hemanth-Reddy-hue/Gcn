import torch
import random
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

def generate_graphs(num_graphs=100):
    graphs = []
    for i in range(num_graphs):
        num_nodes = random.randint(10, 20)
        p = random.uniform(0.1, 0.5)
        G = nx.erdos_renyi_graph(num_nodes, p)
        for node in G.nodes():
            G.nodes[node]['x'] = [G.degree[node]]
        data = from_networkx(G)
        data.x = data.x.float()
        data.y = torch.tensor(random.randint(0, 1))
        graphs.append(data)
    return graphs


class GCNLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        return self.linear(x)
    

def training(Graphs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCNLayer(in_channels=1, hidden_channels=64, out_channels=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.04, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    total_loss = 0
    for i in range(30):
        for graphs in Graphs:
            graphs = graphs.to(device)
            optimizer.zero_grad()
            out = model(graphs.x, graphs.edge_index, graphs.batch)
            loss = loss_fn(out, graphs.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"loss at epoch {i} is {loss.item()}")
    return model, total_loss


def testing(test_data, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_data:
            data = data.to(next(model.parameters()).device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
    return correct / total


graphs = generate_graphs(300)
train_data = DataLoader(graphs[:200], batch_size=16, shuffle=True)
test_data = DataLoader(graphs[200:], batch_size=16)

model, total_loss = training(train_data)
print(f"training finished with a total loss of {total_loss}")

accuracy = testing(test_data, model)
print(f"testing finished with an accuracy of {accuracy}")
