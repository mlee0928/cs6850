import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Define graph data
edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)  # add extra edges
x = torch.tensor([[1.0, 2.0],
                  [2.0, 3.0],
                  [3.0, 4.0],
                  [4.0, 5.0],
                  [5.0, 6.0],
                  [0.0, 0.0],  # add dummy node
                  [0.0, 0.0]], dtype=torch.float)  # add dummy node

y = torch.tensor([0, 1, 1, 0, 1, 0, 0], dtype=torch.long)  # add labels for dummy nodes

# Instantiate model and optimizer
model = GCN(in_channels=2, hidden_channels=16, out_channels=5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train loop
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = F.nll_loss(out[:5], y[:5])  # only compute loss for real nodes
    loss.backward()
    optimizer.step()
    print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss.item()))

# Evaluate on new data
model.eval()
correct = 0
total = 0
x_test = torch.tensor([[0.5, 0.6], [1.0, 1.1], [2.0, 2.1], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float)
y_test = torch.tensor([1, 1, 0, 1, 0])
out_test = model(x_test, edge_index)
pred_test = out_test[:5].argmax(dim=1)  # only predict for real nodes
correct += pred_test.eq(y_test).sum().item()
total += y.size(0)
print('Predictions: {}'.format(pred_test.detach().numpy()))
print(f"Accuracy: {correct / total}")