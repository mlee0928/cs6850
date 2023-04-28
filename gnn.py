import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import json
from sklearn.model_selection import train_test_split
import pdb

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.in_channel = in_channels
        self.hidden_channel = hidden_channels
        self.out_channel = out_channels
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # print(x.shape, edge_index.shape)
        # pdb.set_trace()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        # x = global_mean_pool(x, torch.zeros_like(torch.tensor(batch)))
        return F.log_softmax(x, dim=1)

def loss_fn(gt, pred, output1_w = 0.5, output2_w = 0.5):
    count = gt.shape[0]
    losses = 0
    for i in range(count):
        losses += abs(gt[i][0] - pred[i][0]) * output1_w
        losses += abs(gt[i][1] - pred[i][1]) * output2_w
    losses /= count
    return losses

if torch.cuda:
    device = 'cuda'
else:
    device = 'cpu'
device = 'cpu'
print(f"device: {device}")

with open('data/compile_data.json', 'r', encoding="utf-8") as f:
    compile_data = json.load(f)
    f.close()

edge_key_mapping = dict(zip(compile_data.keys(), range(len(compile_data))))



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

    edge_source = []
    edge_dest = []
    x = []
    y = []
    for vid_id in compile_data.keys():
        source_says_neigh = compile_data[vid_id]["source_neighbors"]
        target_says_neigh = compile_data[vid_id]["target_neighbors"]
        for source in source_says_neigh:
            edge_source.append(edge_key_mapping[source])
            edge_dest.append(edge_key_mapping[vid_id])
        for dest in target_says_neigh:
            edge_dest.append(edge_key_mapping[dest])
            edge_source.append(edge_key_mapping[dest])

        lst = []
        x.append([compile_data[vid_id]["aver_daily_view"], compile_data[vid_id]["aver_daily_share"]])
        # lst.append(compile_data[vid_id]["aver_daily_view"])
        # lst.append(compile_data[vid_id]["aver_daily_share"])
        # lst.append(compile_data[vid_id]["aver_watch_time"])
        # lst.extend(compile_data[vid_id]["neighbor_engagement"])
        # x.append(lst)

        y.append([compile_data[vid_id]["aver_watch_percentage"], compile_data[vid_id]["relative_engagement"]])


    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
    # print(len(x_train))
    x_train = x
    y_train = y
    edge_index = torch.tensor([edge_source, edge_dest], dtype=torch.long).to(device)
    x_train = torch.tensor(x_train, dtype=torch.float).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float).to(device)
    # x_test = torch.tensor(x_test, dtype=torch.float).to(device)
    # y_test = torch.tensor(y_test, dtype=torch.float).to(device)
    x_test = []
    y_test = []

    return edge_index, x_train, y_train, x_test, y_test


edge_index, x_train, y_train, x_test, y_test = get_graph()

# edge_index = torch.tensor([[0, 1, 2, 3, 4],  # source
#                            [1, 2, 3, 4, 0]], # destination
#                           dtype=torch.long).to(device)

# x = torch.tensor([[1.0, 2.0],
#                   [2.0, 3.0],
#                   [3.0, 4.0],
#                   [4.0, 5.0],
#                   [5.0, 6.0],
#                   [0.0, 0.0],
#                   [0.0, 0.0]], dtype=torch.float).to(device)

# x is (5, 2), edge_index is (2, 5)

# y = torch.tensor([[0, 2], [1, 3], [0, 1], [1, 0], [1, 1], [3, 0], [4, 0]], dtype=torch.float).to(device)
print("x_train shape[1]:", x_train.shape[1])
batch = x_train.shape[1]
model = GCN(in_channels=x_train.shape[1], hidden_channels=16, out_channels=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(5):
    optimizer.zero_grad()
    print(x_train.shape, edge_index.shape)
    out = model(x_train, edge_index)
    loss = loss_fn(out, y_train)
    # loss = F.nll_loss(out.view(-1, out.shape[0]).flatten(), y_train.view(-1, y_train.shape[0]).flatten())
    loss.backward()
    optimizer.step()
    print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss.item()))

model.eval()
correct = 0
total = 0
# x_test = torch.tensor([[0.5, 0.6], [1.0, 1.1], [2.0, 2.1], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float).to(device)
# y_test = torch.tensor([[1,1], [1,1], [0,1], [1,1], [0,1]]).to(device)
pred_test = model(x_test, edge_index)
if device == 'cuda':
    pred_test = pred_test.cpu()
    y_test = y_test.cpu()

loss = loss_fn(pred_test, y_test)

print(f"Test Loss: {loss}")