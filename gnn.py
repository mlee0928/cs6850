import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import json
import matplotlib.pyplot as plt
import numpy as np
import random

random.seed(42)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.in_channel = in_channels
        self.hidden_channel = hidden_channels
        self.out_channel = out_channels
        self.conv1 = GCNConv(in_channels, 32)
        self.conv2 = GCNConv(32, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.4)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        return x

def sq_loss_fn(gt, pred, output1_w = 0.5, output2_w = 0.5):
    count = gt.shape[0]
    losses = 0
    for i in range(count):
        losses += ((gt[i][0] - pred[i][0]) * output1_w) ** 2
        losses += ((gt[i][1] - pred[i][1]) * output2_w) ** 2
    losses /= count
    return losses

def abs_loss_fn(gt, pred, output1_w = 0.5, output2_w = 0.5):
    count = gt.shape[0]
    losses = 0
    for i in range(count):
        losses += abs((gt[i][0] - pred[i][0]) * output1_w)
        losses += abs((gt[i][1] - pred[i][1]) * output2_w)
    losses /= count
    return losses

def smape(y_pred, y_test):
    n = len(y_pred)
    acc = 0
    for i in range(n):
        a = abs(y_pred[i][0] - y_test[i][0])
        b = abs(y_pred[i][1] - y_test[i][1])
        c = y_test[i][0] + y_test[i][1]
        d = y_pred[i][0] + y_pred[i][1]
        acc += abs((a + b) / ((c + d) / 2))
    return acc / n

if torch.cuda:
    device = 'cuda'
else:
    device = 'cpu'

print(f"device: {device}")

with open('data/compile_data.json', 'r', encoding="utf-8") as f:
    compile_data = json.load(f)
    f.close()

edge_key_mapping = dict(zip(compile_data.keys(), range(len(compile_data))))

"""
Inputs:  1. neighbor_aver_daily_view
         2. neighbor_aver_daily_share
         3. neighbor_aver_watch_percentage
         4. neighbor_engagement
         5. network centrality
Outputs: 1. aver_watch_percentage
         2. relative_engagement
"""
def get_graph():
    keys = list(edge_key_mapping)
    split_percentages = [0.6, 0.2, 0.2]
    split_lengths = [int(len(keys) * p) for p in split_percentages]
    random.shuffle(keys)

    split1 = keys[:split_lengths[0]]
    split2 = keys[split_lengths[0]:split_lengths[0] + split_lengths[1]]
    split3 = keys[split_lengths[0] + split_lengths[1]:]

    dict1 = {k: edge_key_mapping[k] for k in split1}
    dict2 = {k: edge_key_mapping[k] for k in split2}
    dict3 = {k: edge_key_mapping[k] for k in split3}

    edge_key_mapping1 = dict(zip(dict1.keys(), range(len(dict1))))
    edge_key_mapping2 = dict(zip(dict2.keys(), range(len(dict2))))
    edge_key_mapping3 = dict(zip(dict3.keys(), range(len(dict3))))

    def get_info(dictionary, edge_key_map):
        x = []
        y = []
        edge = [[], []]

        for vid_id in dictionary:
            source_says_neigh = compile_data[vid_id]["source_neighbors"]
            target_says_neigh = compile_data[vid_id]["target_neighbors"]
            for source in source_says_neigh:
                if source in edge_key_map:
                    edge[0].append(edge_key_map[source])
                    edge[1].append(edge_key_map[vid_id])
            for dest in target_says_neigh:
                if dest in edge_key_map:
                    edge[1].append(edge_key_map[vid_id])
                    edge[0].append(edge_key_map[dest])

            all_keys = ["neighbor_aver_daily_view", "neighbor_aver_daily_share",
                        "neighbor_aver_watch_percentage", "neighbor_engagement", "centrality"]
            all_keys = ["neighbor_aver_daily_view", "neighbor_aver_daily_share",
                        "neighbor_engagement", "neighbor_aver_watch_percentage"]
            lst = []
            for k in all_keys:
                lst.extend(compile_data[vid_id][k])

            x.append(lst)
            y.append([compile_data[vid_id]["aver_watch_percentage"], compile_data[vid_id]["relative_engagement"]])

        mag = np.linalg.norm(x)
        x = x / mag

        x = torch.tensor(x, dtype=torch.float).to(device)
        y = torch.tensor(y, dtype=torch.float).to(device)
        edge = torch.tensor(edge, dtype=torch.long).to(device)

        return x, y, edge

    x_train, y_train, edge_train = get_info(dict1, edge_key_mapping1)
    x_val, y_val, edge_val = get_info(dict2, edge_key_mapping2)
    x_test, y_test, edge_test = get_info(dict3, edge_key_mapping3)

    return x_train, x_test, x_val, y_train, y_test, y_val, edge_train, edge_test, edge_val


x_train, x_test, x_val, y_train, y_test, y_val, edge_train, edge_test, edge_val = get_graph()

model = GCN(in_channels=x_train.shape[1], hidden_channels=16, out_channels=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
train_loss = []
val_loss = []

loss_fn = sq_loss_fn

for epoch in range(100):
    optimizer.zero_grad()
    out = model(x_train, edge_train)
    loss = loss_fn(out, y_train)
    train_loss.append(loss.cpu().data.numpy())
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(x_val, edge_val)
        loss = loss_fn(y_val, out)
        val_loss.append(loss.item())
        print('Epoch: {:03d}, Val Loss: {:.4f}'.format(epoch, loss.item()))

        pred_test = model(x_test, edge_test)
        sm = smape(pred_test.cpu().detach().numpy(), y_test.cpu().detach().numpy())

    print(f"SMAPE: {sm}, MSE: {loss}")


plt.plot(train_loss, color='red', label="train loss")
plt.plot(val_loss, color='blue', label="val loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')
num = 3
plt.savefig(f"gnn_loss_plt_{num}.png")

np.savetxt(f"gnn_loss_vals_{num}", np.array([train_loss, val_loss]))

pred_test = model(x_test, edge_test)
if device == 'cuda':
    pred_test = pred_test.cpu()
    y_test = y_test.cpu()

loss = loss_fn(pred_test, y_test)
sm = smape(pred_test.cpu().detach().numpy(), y_test.cpu().detach().numpy())

print(f"SMAPE: {sm}, MSE: {loss}")
