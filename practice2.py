import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ------------------ GCN LAYERS ------------------ #
class GCNLayer(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.linear=nn.Linear(in_features,out_features)

    def forward(self,X,A):
        I = torch.eye(A.size(0))
        A_hat = A + I
        D_hat = torch.diag(torch.pow(torch.sum(A_hat,dim=1),-0.5))
        A_norm=D_hat @ A_hat @ D_hat
        return self.linear(A_norm @ X)

class GCNWithHidden(nn.Module):
    def __init__(self,in_features,hidden_features,out_features):
        super().__init__()
        self.gcn1=GCNLayer(in_features,hidden_features)
        self.gcn2=GCNLayer(hidden_features,out_features)
        self.droupout=nn.Dropout(p=0.5)

    def forward(self,X,A):
        H=F.relu(self.gcn1(X,A))
        H=self.droupout(H)
        return self.gcn2(H,A)

# ------------------ DATASET PREP ------------------ #
dataset=Planetoid(root='/tmp/cora',name='Cora')
data = dataset[0]

def prepare_dataset(data):
    features = data.x
    edges = data.edge_index
    labels = data.y
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    n = data.num_nodes

    adj = torch.zeros((n, n), dtype=torch.float32)
    for i, j in edges.t(): 
        adj[i, j] = 1
        adj[j, i] = 1 

    return features, adj, labels, train_mask, val_mask, test_mask

features, adj, labels, train_mask, val_mask, test_mask = prepare_dataset(data)

# ------------------ VALIDATION ------------------ #
def validate(features, adj, labels, train_mask, val_mask):
    lrs=[0.1,0.01,0.001]
    hds=[8,16,32]
    best_acc=0
    best_params=(0.01,16)

    for lr in lrs:
        for h in hds:
            print("in validation")
            model = GCNWithHidden(features.shape[1],h,labels.max().item()+1)
            optimizer=optim.Adam(model.parameters(),lr=lr,weight_decay=5e-4)
            loss_fn=nn.CrossEntropyLoss()

            for epoch in range(200):
                model.train()
                optimizer.zero_grad()
                out=model(features,adj)
                loss=loss_fn(out[train_mask],labels[train_mask])
                loss.backward()
                optimizer.step()

            # Evaluate on validation
            model.eval()
            with torch.no_grad():
                preds = out[val_mask].argmax(dim=1)
                acc = accuracy_score(labels[val_mask].cpu(), preds.cpu())
                if acc > best_acc:
                    best_acc=acc
                    best_params=(lr,h)
    print(f"Best Val Acc: {best_acc:.4f} with lr={best_params[0]}, hidden={best_params[1]}")
    return best_params

# ------------------ TRAINING ------------------ #
def train(features, adj, labels, train_mask, val_mask, lr, h, epochs=300):
    model = GCNWithHidden(features.shape[1],h,labels.max().item()+1)
    optimizer=optim.Adam(model.parameters(),lr=lr,weight_decay=5e-4)
    loss_fn=nn.CrossEntropyLoss()

    train_losses, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out=model(features,adj)
        loss=loss_fn(out[train_mask],labels[train_mask])
        loss.backward()
        optimizer.step()

        # validation acc
        model.eval()
        with torch.no_grad():
            preds = out[val_mask].argmax(dim=1)
            acc = accuracy_score(labels[val_mask].cpu(), preds.cpu())

        train_losses.append(loss.item())
        val_accs.append(acc)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss {loss.item():.4f}, Val Acc {acc:.4f}")

    return model, train_losses, val_accs

# ------------------ TESTING ------------------ #
def test(model, features, adj, labels, test_mask):
    model.eval()
    with torch.no_grad():
        out = model(features,adj)
        preds = out[test_mask].argmax(dim=1)
        acc = accuracy_score(labels[test_mask].cpu(), preds.cpu())
        f1 = f1_score(labels[test_mask].cpu(), preds.cpu(), average="weighted")
        print("Test Accuracy:", acc)
        print("Test F1 Score:", f1)
        print(classification_report(labels[test_mask].cpu(), preds.cpu()))
    return acc, f1

# ------------------ RUN PIPELINE ------------------ #
print("started")
best_lr, best_h = validate(features,adj,labels,train_mask,val_mask)
print("validation done")
model, train_losses, val_accs = train(features,adj,labels,train_mask,val_mask,best_lr,best_h)
print("training done")
acc, f1 = test(model,features,adj,labels,test_mask)
print("testing done")


# ------------------ PLOTS ------------------ #
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_losses,label="Train Loss")
plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.legend()

plt.subplot(1,2,2)
plt.plot(val_accs,label="Validation Accuracy")
plt.xlabel("Epochs"); plt.ylabel("Accuracy"); plt.legend()
plt.show()
