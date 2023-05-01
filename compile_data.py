import json
import pickle
import isodate
import os
import numpy as np
import networkx as nx
import torch

class EngagementMap(object):
    # https://github.com/avalanchesiqi/youtube-engagement/blob/master/engagement_map/query_engagement.py
    def __init__(self, engagement_path):
        self.engagement_map = None
        self.duration_axis = None
        self.bin_size = None
        self.load_engagement_map(engagement_path)

    def load_engagement_map(self, engagement_filepath):
        """ Load engagement map.
        """
        if not os.path.exists(engagement_filepath):
            raise Exception('no engagement file is found!')

        self.engagement_map = pickle.load(open(engagement_filepath, 'rb'))
        self.duration_axis = self.engagement_map['duration']
        self.bin_size = len(self.engagement_map[self.engagement_map['duration'][0]])

    def query_engagement_map(self, duration, watch_percentage):
        """ Query the engagement map for relative engagement given video length and watch percentage.
        """
        try:
            bin_x_idx = next(idx for idx, length in enumerate(self.duration_axis) if length >= duration)
        except StopIteration:
            bin_x_idx = len(self.duration_axis) - 1

        correspond_watch_percentage = self.engagement_map[bin_x_idx]
        try:
            relative_engagement = next(y for y, val in enumerate(correspond_watch_percentage) if val > watch_percentage) / self.bin_size
        except StopIteration:
            relative_engagement = 1
        return relative_engagement


"""
Inputs: 1. average daily view q
        2. average daily share s
        3. average daily watch time w
        4. the node's neighbors’ relative engagement H for each video v, where H is a vector of the relative
           engagement η for v’s 20 neighbors (where each neighbor is a video present in v’s relevant list). 
Outputs: 1. average watch percentage pv
         2. relative engagement ηv.
"""


"""
# uncomment to compile data from vevo_en_videos_60k.json
all_data = {}
filename = "vevo_en_videos_60k"
for line in open(f'data/{filename}.json', 'r', encoding="utf-8"):
    data = json.loads(line)
    vid_id = data.pop("id")
    # print(vid_id)
    # print(data.keys())
    all_data[vid_id] = data

with open("data/all_data.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f)
    f.close()
"""

print("loading a bunch of data...")
with open('data/engagement_map.p', 'rb') as f:
    engagement_map = pickle.load(f)
    f.close()
print("part 1 engagement map done...")

# print(engagement_map["duration"][100], engagement_map[110])

with open("data/all_data.json", "r", encoding="utf-8") as f:
    all_data = json.load(f)
print("part 2 all_data.json done...")

with open("output.json", "r", encoding="utf-8") as f:
    output_data = json.load(f)
print("part 3 output.json done...\n")

print("checking everything in output.json is in all_data.json...")
count = 0
for vid_id in output_data:
    if vid_id not in all_data:
        # print(vid_id)
        count += 1
print(f"bad ones: {count}, total: {len(all_data)}, percentage bad: {float(count) / len(all_data)}\n")

print("checking the opposite...")
count = 0
for vid_id in all_data:
    if vid_id not in output_data:
        # print(vid_id)
        count += 1
print(f"bad ones: {count}, total: {len(all_data)}, percentage bad: {float(count) / len(all_data)}\n")

engagement = EngagementMap('data/engagement_map.p')

compile_data = {}
max_neighbors = 0
for vid_id in output_data:
    vid_dict = output_data[vid_id]
    all_data_dict = all_data[vid_id]
    eid = vid_dict["eid"]
    neighbors = []
    source_neighbors = []
    target_neighbors = []
    for nei in vid_dict["source_neighbors"]:
        if nei in output_data:
            source_neighbors.append(nei)
    for nei in vid_dict["target_neighbors"]:
        if nei in output_data:
            target_neighbors.append(nei)
    neighbors.extend(source_neighbors)
    neighbors.extend(target_neighbors)
    max_neighbors = max(max_neighbors, len(neighbors))
    aver_daily_view = vid_dict["average_daily_view"]
    aver_daily_share = vid_dict["average_daily_share"]
    duration = all_data_dict['contentDetails']['duration']
    duration = isodate.parse_duration(duration)
    duration = duration.total_seconds()
    if duration == 0.0:
        continue
    total_view = all_data_dict['insights']['totalView']
    total_watch_time = all_data_dict['insights']['avgWatch'] * len(all_data_dict['insights']['dailyWatch'])
    aver_watch_time = all_data_dict['insights']['avgWatch']
    aver_watch_percentage = aver_watch_time / duration * 100
    relative_engagement = engagement.query_engagement_map(duration, aver_watch_percentage)
    # print(relative_engagement)
    compile_data[vid_id] = {"eid": eid, "neighbors": neighbors, "aver_daily_share": aver_daily_share,
                            "duration": duration, "total_view": total_view, "total_watch_time": total_watch_time,
                            "aver_watch_time": aver_watch_time, "aver_watch_percentage": aver_watch_percentage,
                            "relative_engagement": relative_engagement, "aver_daily_view": aver_daily_view,
                            "source_neighbors":  source_neighbors,
                            "target_neighbors":  target_neighbors}

number_empty = 0
for vid_id in compile_data:
    neighbors = compile_data[vid_id]["neighbors"]
    if len(neighbors) == 0:
        number_empty += 1
    neighbor_engagement = [
        engagement.query_engagement_map(compile_data[item]["duration"],
                                        compile_data[item]["aver_watch_percentage"]) for item in neighbors
    ]
    neighbor_engagement = np.pad(neighbor_engagement, (0, max_neighbors - len(neighbor_engagement)), 'constant', constant_values=0)
    neighbor_engagement = neighbor_engagement.tolist()
    compile_data[vid_id]["neighbor_engagement"] = neighbor_engagement
            
    neighbor_aver_daily_view = [compile_data[item]["aver_daily_view"] for item in neighbors]
    neighbor_aver_daily_share = [compile_data[item]["aver_daily_share"] for item in neighbors]
    neighbor_aver_watch_time = [compile_data[item]["aver_watch_time"] for item in neighbors]
    neighbor_aver_watch_percentage = [compile_data[item]["aver_watch_percentage"] for item in neighbors]
    
    neighbor_aver_daily_view = np.pad(neighbor_aver_daily_view, (0, max_neighbors - len(neighbor_aver_daily_view)), 'constant', constant_values=0)
    neighbor_aver_daily_share = np.pad(neighbor_aver_daily_share, (0, max_neighbors - len(neighbor_aver_daily_share)), 'constant', constant_values=0)
    neighbor_aver_watch_time = np.pad(neighbor_aver_watch_time, (0, max_neighbors - len(neighbor_aver_watch_time)), 'constant', constant_values=0)
    neighbor_aver_watch_percentage = np.pad(neighbor_aver_watch_percentage, (0, max_neighbors - len(neighbor_aver_watch_percentage)), 'constant', constant_values=0)
    
    compile_data[vid_id]["neighbor_aver_daily_view"] = neighbor_aver_daily_view.tolist()
    compile_data[vid_id]["neighbor_aver_daily_share"] = neighbor_aver_daily_share.tolist()
    compile_data[vid_id]["neighbor_aver_watch_time"] = neighbor_aver_watch_time.tolist()
    compile_data[vid_id]["neighbor_aver_watch_percentage"] = neighbor_aver_watch_percentage.tolist()
    
print(f'Percentage of videos with neighbors: {number_empty/len(compile_data)}')

edge_key_mapping = dict(zip(compile_data.keys(), range(len(compile_data))))
keys = list(edge_key_mapping)

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

graph = get_networkx()
centrality = nx.eigenvector_centrality_numpy(graph)

for vid_id in compile_data:
    central = []
    for vid in compile_data[vid_id]["neighbors"]:
        central.append(centrality[edge_key_mapping[vid]])
    central = np.pad(central, (0, len(compile_data[vid_id]["neighbor_engagement"]) - len(central)), 'constant',
                        constant_values=0)
    central = central.tolist()
    compile_data[vid_id]['centrality'] = central

        
with open("data/compile_data.json", "w", encoding="utf-8") as f:
    json.dump(compile_data, f)


