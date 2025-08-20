import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class GCNLayer(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.linear = nn.Linear(in_features,out_features)
    def forward(self,X,A):
        I = torch.eye(A.size(0))
        A_hat = A + I
        D_hat = torch.diag(torch.sum(A_hat,dim=1)**-0.5)
        A_norm=D_hat @ A_hat @ D_hat
        return F.relu(self.linear(A_norm@X))
    
X = torch.rand(4,2)
A=torch.tensor([[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]],dtype=torch.float32)

y = torch.tensor([0,1,0,1],dtype=torch.long)
class GCN(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.gcn=GCNLayer(in_features,out_features)
    def forward(self,X,A):
        return self.gcn(X,A)

class GCNWithhidden(nn.Module):
    def __init__(self,in_features,hidden_fetaures,out_features):
        super().__init__()
        self.gcn1=GCNLayer(in_features,hidden_fetaures)
        self.gcn2=GCNLayer(hidden_fetaures,out_features)

    def   forward(self,X,A):
        H=F.relu(self.gcn1(X,A))
        return self.gcn2(H,A)
      
model = GCN(2,2)
optimizer = optim.Adam(model.parameters(),lr=0.4)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(100):
    optimizer.zero_grad()
    out = model(X,A)
    loss = loss_fn(out,y)
    loss.backward()
    optimizer.step()
    if epoch % 20 :
        print(f" loss at epoch {epoch} is {loss}")

pred = torch.argmax(model(X,A),dim=1)
print("predicted classes ")
print(pred)


print("with hidden layers")
model = GCNWithhidden(2,4,2)
optimizer=optim.Adam(model.parameters(),lr=0.4)
for epoch in range(100):
    optimizer.zero_grad()
    out=model(X,A)
    loss = loss_fn(out,y)
    loss.backward()
    optimizer.step()
    if epoch % 20:
        print(f"loss at epoch {epoch} with hidden layer is {loss}")

pred = torch.argmax(model(X,A),dim=1)
n=y.shape[0]
for i in range(n):
    print(f" actual is {y[i]} :: prediction is {pred[i]}")
