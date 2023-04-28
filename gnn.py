import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import json

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

if torch.cuda:
    device = 'cuda'
else:
    device = 'cpu'
print(f"device: {device}")

with open('data/compile_data.json', 'r', encoding="utf-8") as f:
    compile_data = json.load(f)

edge_map = dict(zip(range(len(compile_data)), compile_data.keys()))
print(edge_map)

"""
Inputs:  1. aver_daily_view
         2. aver_daily_share
         3. aver_watch_time
         4. neighbor_engagement
Outputs: 1. aver_watch_percentage
         2. relative_engagement
"""
# TODO: construct node values with options (for both x and y)
def get_graph(input1=True, input2=True, input3=True, input4=True, output1=True, output2=True):
    if input1 + input2 + input3 + input4 == 0 or output1 + output2 == 0:
        raise Exception("Error: Either no choice of input (x options) or output (y options)")



edge_index = torch.tensor([[0, 1, 2, 3, 4],  # source
                           [1, 2, 3, 4, 0]], # destination
                          dtype=torch.long).to(device)
x = torch.tensor([[1.0, 2.0],
                  [2.0, 3.0],
                  [3.0, 4.0],
                  [4.0, 5.0],
                  [5.0, 6.0],
                  [0.0, 0.0],
                  [0.0, 0.0]], dtype=torch.float).to(device)

y = torch.tensor([0, 1, 1, 0, 1, 0, 0], dtype=torch.long).to(device)

model = GCN(in_channels=2, hidden_channels=16, out_channels=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(50):
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = F.nll_loss(out[:5], y[:5])
    loss.backward()
    optimizer.step()
    print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss.item()))

model.eval()
correct = 0
total = 0
x_test = torch.tensor([[0.5, 0.6], [1.0, 1.1], [2.0, 2.1], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float).to(device)
y_test = torch.tensor([1, 1, 0, 1, 0]).to(device)
out_test = model(x_test, edge_index)
pred_test = out_test[:5].argmax(dim=1)
if device == 'cuda':
    pred_test = pred_test.cpu()
    y_test = y_test.cpu()
correct += pred_test.eq(y_test).sum().item()
total += y.size(0)
print('Predictions: {}'.format(pred_test.detach().numpy()))
print(f"Accuracy: {correct / total}")