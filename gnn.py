import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

# from deepsnap.graph import Graph
#
# from sklearn.model_selection import train_test_split
# from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit
# from torch_geometric.utils import subgraph
# from deepsnap.dataset import GraphDataset
import networkx as nx
import random


# import copy
#
# import pdb
# from torch_geometric.data import Data

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
        # print(x.shape, edge_index.shape)
        # pdb.set_trace()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.4)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        # x = global_mean_pool(x, torch.zeros_like(torch.tensor(batch)))
        return x #F.log_softmax(x, dim=1)

def loss_fn(gt, pred, output1_w = 0.5, output2_w = 0.5):
    count = gt.shape[0]
    # print("gt shape:", count)
    losses = 0
    for i in range(count):
        losses += ((gt[i][0] - pred[i][0]) * output1_w) ** 2
        losses += ((gt[i][1] - pred[i][1]) * output2_w) ** 2
    losses /= count
    return losses

if torch.cuda:
    device = 'cuda'
else:
    device = 'cpu'

print(f"device: {device}")

with open('data/compile_data.json', 'r', encoding="utf-8") as f:
    compile_data = json.load(f)
    f.close()

edge_key_mapping = dict(zip(compile_data.keys(), range(len(compile_data))))

random_keys = random.sample(list(edge_key_mapping), int(len(edge_key_mapping) * 0.8))

# Split the dictionary into two based on the random keys
dict1 = {k: edge_key_mapping[k] for k in random_keys}
dict2 = {k: edge_key_mapping[k] for k in edge_key_mapping if k not in random_keys}


"""
Inputs:  1. aver_daily_view
         2. aver_daily_share
         3. aver_watch_time
         4. neighbor_engagement
         5. network centrality
Outputs: 1. aver_watch_percentage
         2. relative_engagement
"""
def get_networkx():
    G = nx.Graph()

    for vid_id in compile_data.keys():
        lst = []
        lst.append(compile_data[vid_id]["aver_daily_view"])
        lst.append(compile_data[vid_id]["aver_daily_share"])
        lst.append(compile_data[vid_id]["aver_watch_time"])
        lst.extend(compile_data[vid_id]["neighbor_engagement"])
        lst.extend([compile_data[vid_id]["aver_watch_percentage"], compile_data[vid_id]["relative_engagement"]])
        G.add_node(edge_key_mapping[vid_id], node_feature=torch.tensor(lst, dtype=float))

    for vid_id in compile_data.keys():

        source_says_neigh = compile_data[vid_id]["source_neighbors"]
        target_says_neigh = compile_data[vid_id]["target_neighbors"]
        for source in source_says_neigh:
            G.add_edge(edge_key_mapping[source], edge_key_mapping[vid_id])
        for dest in target_says_neigh:
            G.add_edge(edge_key_mapping[dest], edge_key_mapping[vid_id])

    return G

def get_graph(input1=True, input2=True, input3=True, input4=True, output1=True, output2=True):
    if input1 + input2 + input3 + input4 == 0 or output1 + output2 == 0:
        raise Exception("Error: Either no choice of input (x options) or output (y options)")

    # G = nx.Graph()


    # edge_source = []
    # edge_dest = []
    # x = []
    # y = []
    # x_y = []
    # for vid_id in compile_data.keys():
    #     lst = []
    #     lst.append(compile_data[vid_id]["aver_daily_view"])
    #     lst.append(compile_data[vid_id]["aver_daily_share"])
    #     lst.append(compile_data[vid_id]["aver_watch_time"])
    #     lst.extend(compile_data[vid_id]["neighbor_engagement"])
    #     lst.extend([compile_data[vid_id]["aver_watch_percentage"], compile_data[vid_id]["relative_engagement"]])
    #     G.add_node(edge_key_mapping[vid_id], node_feature=torch.tensor(lst, dtype=float))
    x_train = []
    y_train = []
    edge_train = [[], []]
    dict1_nodes = defaultdict(bool)
    for vid_id in dict1:
        dict1_nodes[vid_id] = True
        for node in compile_data[vid_id]["neighbors"]:
            dict1_nodes[node] = True
    for vid_id in dict1_nodes: #compile_data.keys():

        source_says_neigh = compile_data[vid_id]["source_neighbors"]
        target_says_neigh = compile_data[vid_id]["target_neighbors"]
        for source in source_says_neigh:
            edge_train[0].append(edge_key_mapping[source])
            edge_train[1].append(edge_key_mapping[vid_id])
            # G.add_edge(edge_key_mapping[source], edge_key_mapping[vid_id])
        for dest in target_says_neigh:
            edge_train[1].append(edge_key_mapping[vid_id])
            edge_train[0].append(edge_key_mapping[dest])
            # G.add_edge(edge_key_mapping[dest], edge_key_mapping[vid_id])

        lst = []
        # x.append([compile_data[vid_id]["aver_daily_view"], compile_data[vid_id]["aver_daily_share"]])
        lst.append(compile_data[vid_id]["aver_daily_view"])
        lst.append(compile_data[vid_id]["aver_daily_share"])
        lst.append(compile_data[vid_id]["aver_watch_time"])
        lst.extend(compile_data[vid_id]["neighbor_engagement"])
        x_train.append(lst)

        # lst.extend([compile_data[vid_id]["aver_watch_percentage"], compile_data[vid_id]["relative_engagement"]])
        y_train.append([compile_data[vid_id]["aver_watch_percentage"], compile_data[vid_id]["relative_engagement"]])
        # x_y.append(lst)

    x_test = []
    y_test = []
    edge_test = [[], []]
    dict2_nodes = defaultdict(bool)
    for vid_id in dict2:
        dict2_nodes[vid_id] = True
        for node in compile_data[vid_id]["neighbors"]:
            dict2_nodes[node] = True
    for vid_id in dict2_nodes:  # compile_data.keys():

        source_says_neigh = compile_data[vid_id]["source_neighbors"]
        target_says_neigh = compile_data[vid_id]["target_neighbors"]
        for source in source_says_neigh:
            edge_test[0].append(edge_key_mapping[source])
            edge_test[1].append(edge_key_mapping[vid_id])
            # G.add_edge(edge_key_mapping[source], edge_key_mapping[vid_id])
        for dest in target_says_neigh:
            edge_test[1].append(edge_key_mapping[vid_id])
            edge_test[0].append(edge_key_mapping[dest])
            # G.add_edge(edge_key_mapping[dest], edge_key_mapping[vid_id])

        lst = []
        # x.append([compile_data[vid_id]["aver_daily_view"], compile_data[vid_id]["aver_daily_share"]])
        lst.append(compile_data[vid_id]["aver_daily_view"])
        lst.append(compile_data[vid_id]["aver_daily_share"])
        lst.append(compile_data[vid_id]["aver_watch_time"])
        lst.extend(compile_data[vid_id]["neighbor_engagement"])
        x_test.append(lst)

        # lst.extend([compile_data[vid_id]["aver_watch_percentage"], compile_data[vid_id]["relative_engagement"]])
        y_test.append([compile_data[vid_id]["aver_watch_percentage"], compile_data[vid_id]["relative_engagement"]])
        # x_y.append(lst)


    # edge_index = torch.tensor([edge_source, edge_dest], dtype=torch.long).to(device)
    # x = torch.tensor(x, dtype=torch.float).to(device)
    # y = torch.tensor(y, dtype=torch.float).to(device)
    # x_y = torch.tensor(x_y, dtype=float).to(device)
    # print(x_y.shape, edge_index.shape)
    # x_train, x_test = train_test_split(x_y, test_size=0.2, random_state=42)
    # y_train = x_train[:, -2:]
    # x_train = x_train[:, :4]
    # y_test = x_test[:, -2:]
    # x_test = x_test[:, :4]

    # for vid_id in compile_data.keys():
    #     source_says_neigh = compile_data[vid_id]["source_neighbors"]
    #     target_says_neigh = compile_data[vid_id]["target_neighbors"]
    #     for source in source_says_neigh:
    #         edge_source.append(edge_key_mapping[source])
    #         edge_dest.append(edge_key_mapping[vid_id])
    #         G.add_edge(edge_key_mapping[source], edge_key_mapping[vid_id])
    #     for dest in target_says_neigh:
    #         edge_dest.append(edge_key_mapping[vid_id])
    #         edge_source.append(edge_key_mapping[dest])
    #         G.add_edge(edge_key_mapping[dest], edge_key_mapping[vid_id])
    # data = Data(x=x, y=y, edge_index=edge_index)
    #
    # train_data = Data(x=x_y,
    #                   edge_index=edge_index,
    #                   num_classes=len(lst),
    #                   )

    # nx_graph = data.to_networkx()
    # G = Graph(nx_graph)
    # print(data.x)
    # dataset = GraphDataset(graphs=[G], task='node')
    # print(dataset)
    # train_data, val_data, test_data = dataset.split(transductive=True, split_ratio=[0.8, 0.1, 0.1])
    # print(train_data, val_data, test_data)

    # train_data, val_data, t_est_data = tfs(data)

    # train_mask = torch.rand(len(compile_data)) < 0.6
    # test_mask = ~train_mask
    #
    # train_data = copy.copy(data)
    # train_data.edge_index, _ = subgraph(train_mask, data.edge_index, relabel_nodes=True)
    # train_data.x = data.x[train_mask]
    # train_data.y = data.y[train_mask]
    #
    # test_data = copy.copy(data)
    # test_data.edge_index, _ = subgraph(test_mask, data.edge_index, relabel_nodes=True)
    # test_data.x = data.x[test_mask]
    # test_data.y = data.y[test_mask]
    #
    # val_mask = torch.rand(test_mask.num_nodes) < 0.5
    # test_mask = ~val_mask
    #
    # val_data = copy.copy(data)
    # val_data.edge_index, _ = subgraph(val_mask, data.edge_index, relabel_nodes=True)
    # val_data.x = data.x[val_mask]
    # val_data.y = data.y[val_mask]
    #
    #
    # test_data = copy.copy(data)
    # test_data.edge_index, _ = subgraph(test_mask, data.edge_index, relabel_nodes=True)
    # test_data.x = data.x[test_mask]
    # test_data.y = data.y[test_mask]
    x_train = torch.tensor(x_train, dtype=torch.float).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float).to(device)
    edge_train = torch.tensor(edge_train, dtype=torch.long).to(device)
    edge_test = torch.tensor(edge_test, dtype=torch.long).to(device)

    return x_train, x_test, y_train, y_test, edge_train, edge_test
    # return train_data.x, train_data.y, train_data.edge_index,\
    #     val_data.x, val_data.y, val_data.edge_index,\
    #     test_data.x, test_data.y, test_data.edge_index

used = {}
def get_graph2():
    num_nodes = 0
    total_nodes = 35835

    visited = {}
    for node in compile_data.keys():
        q = []
        if node in visited:
            continue
        q.append(node)
        visited[node] = True
        while q:
            curr = q.pop(0)
            for neighbor in compile_data[curr]["neighbors"]:
                if neighbor not in visited:
                    q.append(neighbor)
                    visited[neighbor] = True
        # print(len(visited) / total_nodes * 100)
        if len(visited) / total_nodes * 100 > 70:
            break
    # print(len(visited) / total_nodes * 100)

    dict1 = visited
    dict2 = {}
    for node in compile_data.keys():
        if node not in dict1:
            dict2[node] = True

    edge_key_mapping1 = dict(zip(dict1.keys(), range(len(dict1))))
    edge_key_mapping2 = dict(zip(dict2.keys(), range(len(dict2))))

    x_test = []
    y_test = []
    edge_test = [[], []]
    edge_key_mapping = edge_key_mapping2
    for vid_id in dict2:
        source_says_neigh = compile_data[vid_id]["source_neighbors"]
        target_says_neigh = compile_data[vid_id]["target_neighbors"]
        for source in source_says_neigh:
            if source in edge_key_mapping2:
                edge_test[0].append(edge_key_mapping[source])
                edge_test[1].append(edge_key_mapping[vid_id])
        for dest in target_says_neigh:
            if dest in edge_key_mapping2:
                edge_test[1].append(edge_key_mapping[vid_id])
                edge_test[0].append(edge_key_mapping[dest])

        lst = []
        lst.append(compile_data[vid_id]["aver_daily_view"])
        lst.append(compile_data[vid_id]["aver_daily_share"])
        lst.append(compile_data[vid_id]["aver_watch_time"])
        lst.extend(compile_data[vid_id]["neighbor_engagement"])
        x_test.append(lst)

        y_test.append([compile_data[vid_id]["aver_watch_percentage"], compile_data[vid_id]["relative_engagement"]])

    x_train = []
    y_train = []
    edge_train = [[], []]
    edge_key_mapping = edge_key_mapping1
    for vid_id in dict1:
        source_says_neigh = compile_data[vid_id]["source_neighbors"]
        target_says_neigh = compile_data[vid_id]["target_neighbors"]
        for source in source_says_neigh:
            edge_train[0].append(edge_key_mapping[source])
            edge_train[1].append(edge_key_mapping[vid_id])
        for dest in target_says_neigh:
            edge_train[1].append(edge_key_mapping[vid_id])
            edge_train[0].append(edge_key_mapping[dest])

        lst = []
        lst.append(compile_data[vid_id]["aver_daily_view"])
        lst.append(compile_data[vid_id]["aver_daily_share"])
        lst.append(compile_data[vid_id]["aver_watch_time"])
        lst.extend(compile_data[vid_id]["neighbor_engagement"])
        x_train.append(lst)

        y_train.append([compile_data[vid_id]["aver_watch_percentage"], compile_data[vid_id]["relative_engagement"]])

    x_train = torch.tensor(x_train, dtype=torch.float).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float).to(device)
    edge_train = torch.tensor(edge_train, dtype=torch.long).to(device)
    edge_test = torch.tensor(edge_test, dtype=torch.long).to(device)

    return x_train, x_test, y_train, y_test, edge_train, edge_test

def get_graph3():
    graph = get_networkx()
    centrality = nx.eigenvector_centrality_numpy(graph)

    edge_key_mapping = dict(zip(compile_data.keys(), range(len(compile_data))))
    # random_keys = random.sample(list(edge_key_mapping), int(len(edge_key_mapping) * 0.7))

    keys = list(edge_key_mapping)
    split_percentages = [0.8, 0.1, 0.1]
    split_lengths = [int(len(keys) * p) for p in split_percentages]
    random.shuffle(keys)

    split1 = keys[:split_lengths[0]]
    split2 = keys[split_lengths[0]:split_lengths[0] + split_lengths[1]]
    split3 = keys[split_lengths[0] + split_lengths[1]:]

    dict1 = {k: edge_key_mapping[k] for k in split1}
    dict2 = {k: edge_key_mapping[k] for k in split2}
    dict3 = {k: edge_key_mapping[k] for k in split3}

    # print(f"total dict: {len(compile_data)}, dict1 size: {len(dict1)}, dict2 size: {len(dict2)}, dict3 size: {len(dict3)}")

    edge_key_mapping1 = dict(zip(dict1.keys(), range(len(dict1))))
    edge_key_mapping2 = dict(zip(dict2.keys(), range(len(dict2))))
    edge_key_mapping3 = dict(zip(dict3.keys(), range(len(dict3))))

    g1 = nx.Graph()
    g2 = nx.Graph()
    g3 = nx.Graph()

    x_train = []
    y_train = []
    edge_train = [[], []]
    neighbors = []
    for vid_id in dict1:
        source_says_neigh = compile_data[vid_id]["source_neighbors"]
        target_says_neigh = compile_data[vid_id]["target_neighbors"]
        for source in source_says_neigh:
            if source in edge_key_mapping1:
                edge_train[0].append(edge_key_mapping1[source])
                edge_train[1].append(edge_key_mapping1[vid_id])
                g1.add_edge(edge_key_mapping1[source], edge_key_mapping1[vid_id])
        for dest in target_says_neigh:
            if dest in edge_key_mapping1:
                edge_train[1].append(edge_key_mapping1[vid_id])
                edge_train[0].append(edge_key_mapping1[dest])
                g1.add_edge(edge_key_mapping1[dest], edge_key_mapping1[vid_id])

        lst = []
        lst.append(compile_data[vid_id]["aver_daily_view"])
        lst.append(compile_data[vid_id]["aver_daily_share"])
        lst.append(compile_data[vid_id]["aver_watch_time"])
        lst.extend(compile_data[vid_id]["neighbor_engagement"])
        #
        # central = []
        # for vid in compile_data[vid_id]["neighbors"]:
        #     central.append(centrality[edge_key_mapping[vid]])
        # central = np.pad(central, (0, len(compile_data[vid_id]["neighbor_engagement"]) - len(central)), 'constant',
        #                  constant_values=0)
        # central = central.tolist()
        # lst.extend(central)

        x_train.append(lst)

        y_train.append([compile_data[vid_id]["aver_watch_percentage"], compile_data[vid_id]["relative_engagement"]])





    x_val = []
    y_val = []
    edge_val = [[], []]
    neighbors = []
    for vid_id in dict2:
        source_says_neigh = compile_data[vid_id]["source_neighbors"]
        target_says_neigh = compile_data[vid_id]["target_neighbors"]
        for source in source_says_neigh:
            if source in edge_key_mapping2:
                edge_val[0].append(edge_key_mapping2[source])
                edge_val[1].append(edge_key_mapping2[vid_id])
                g2.add_edge(edge_key_mapping2[source], edge_key_mapping2[vid_id])
                neighbors.append(source)
        for dest in target_says_neigh:
            if dest in edge_key_mapping2:
                edge_val[1].append(edge_key_mapping2[vid_id])
                edge_val[0].append(edge_key_mapping2[dest])
                g2.add_edge(edge_key_mapping2[dest], edge_key_mapping2[vid_id])
                neighbors.append(dest)

        lst = []
        lst.append(compile_data[vid_id]["aver_daily_view"])
        lst.append(compile_data[vid_id]["aver_daily_share"])
        lst.append(compile_data[vid_id]["aver_watch_time"])
        lst.extend(compile_data[vid_id]["neighbor_engagement"])
        #
        # central = []
        # for vid in compile_data[vid_id]["neighbors"]:
        #     central.append(centrality[edge_key_mapping[vid]])
        # central = np.pad(central, (0, len(compile_data[vid_id]["neighbor_engagement"]) - len(central)), 'constant',
        #                  constant_values=0)
        # central = central.tolist()
        # lst.extend(central)
        x_val.append(lst)

        y_val.append([compile_data[vid_id]["aver_watch_percentage"], compile_data[vid_id]["relative_engagement"]])




    x_test = []
    y_test = []
    edge_test = [[], []]
    neighbors = []
    for vid_id in dict3:
        source_says_neigh = compile_data[vid_id]["source_neighbors"]
        target_says_neigh = compile_data[vid_id]["target_neighbors"]
        for source in source_says_neigh:
            if source in edge_key_mapping3:
                edge_test[0].append(edge_key_mapping3[source])
                edge_test[1].append(edge_key_mapping3[vid_id])
                g3.add_edge(edge_key_mapping3[source], edge_key_mapping3[vid_id])
                neighbors.append(source)
        for dest in target_says_neigh:
            if dest in edge_key_mapping3:
                edge_test[1].append(edge_key_mapping3[vid_id])
                edge_test[0].append(edge_key_mapping3[dest])
                g3.add_edge(edge_key_mapping3[dest], edge_key_mapping3[vid_id])
                neighbors.append(dest)

        lst = []
        lst.append(compile_data[vid_id]["aver_daily_view"])
        lst.append(compile_data[vid_id]["aver_daily_share"])
        lst.append(compile_data[vid_id]["aver_watch_time"])
        lst.extend(compile_data[vid_id]["neighbor_engagement"])
        # central = []
        # for vid in compile_data[vid_id]["neighbors"]:
        #     central.append(centrality[edge_key_mapping[vid]])
        # central = np.pad(central, (0, len(compile_data[vid_id]["neighbor_engagement"]) - len(central)), 'constant',
        #                  constant_values=0)
        # central = central.tolist()
        # lst.extend(central)
        x_test.append(lst)

        y_test.append([compile_data[vid_id]["aver_watch_percentage"], compile_data[vid_id]["relative_engagement"]])



    x_train = torch.tensor(x_train, dtype=torch.float).to(device)
    x_val = torch.tensor(x_val, dtype=torch.float).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float).to(device)
    edge_train = torch.tensor(edge_train, dtype=torch.long).to(device)
    edge_val = torch.tensor(edge_val, dtype=torch.long).to(device)
    edge_test = torch.tensor(edge_test, dtype=torch.long).to(device)

    return x_train, x_test, x_val, y_train, y_test, y_val, edge_train, edge_test, edge_val


# print(centrality)


# x_train, y_train, edge_index_train,\
#     x_val, y_val, edge_index_val, \
#     x_test, y_test, edge_index_test = get_graph()

x_train, x_test, x_val, y_train, y_test, y_val, edge_train, edge_test, edge_val = get_graph3()
# x_train, x_test, y_train, y_test, edge_train, edge_test = get_graph()

# print("x_train shape[1]:", x_train.shape[1])
# batch = x_train.shape[1]
model = GCN(in_channels=x_train.shape[1], hidden_channels=16, out_channels=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
train_loss = []
val_loss = []
# best_val_loss = float('inf')
# patience = 10
# counter = 0
def smape(y_pred, y_test):
    print(y_pred[0])
    n = len(y_pred)
    print(n)
    acc = 0
    for i in range(n):
        a = abs(y_pred[i][0] - y_test[i][0])
        b = abs(y_pred[i][1] - y_test[i][1])
        c = y_test[i][0] + y_test[i][1]
        d = y_pred[i][0] + y_pred[i][1]
        acc += abs((a + b) / ((c + d) / 2))
    return acc / n

for epoch in range(150):
    optimizer.zero_grad()
    # print(x_train.shape, edge_train.shape)
    out = model(x_train, edge_train)
    loss = loss_fn(out, y_train)
    if epoch > 50:
        train_loss.append(loss.cpu().data.numpy())
    # loss = F.nll_loss(out.view(-1, out.shape[0]).flatten(), y_train.view(-1, y_train.shape[0]).flatten())
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(x_val, edge_val)
        loss = loss_fn(y_val, out)
        if epoch > 50:
            val_loss.append(loss.item())
        print('Epoch: {:03d}, Val Loss: {:.4f}'.format(epoch, loss.item()))

        pred_test = model(x_test, edge_test)
        print(type(pred_test.cpu().detach().numpy()))
        print(type(y_test.cpu().detach().numpy()))
        sm = smape(pred_test.cpu().detach().numpy(), y_test.cpu().detach().numpy())

    print(f"SMAPE: {sm}, MSE: {loss}")
    # print(loss.item(), type(loss.item()), best_val_loss, type(best_val_loss))
    # if loss.item() < best_val_loss:
    #     best_val_loss = loss.item()
    #     counter = 0
    # else:
    #     counter += 1

    # if counter >= patience:
    #     print("Early stopping after {} epochs".format(epoch))
    #     break

plt.plot(train_loss, color='red', label="train loss")
plt.plot(val_loss, color='blue', label="val loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss (starting at epoch 50)')
plt.savefig("gnn_loss_plt.png")

pred_test = model(x_test, edge_test)
if device == 'cuda':
    pred_test = pred_test.cpu()
    y_test = y_test.cpu()

r2 = r2_score(pred_test.detach().numpy(), y_test.detach().numpy())
loss = loss_fn(pred_test, y_test)

print(f"R^2: {r2}, MSE: {loss}")
