import csv
import json
import isodate

network = csv.DictReader(open("./data/persistent_network.csv"))
videos_info = []

with open("./data/vevo_en_videos_60k.json") as f:
  for jsonObj in f:
    videos_info.append(json.loads(jsonObj))

output = {}
'''
format:
  {
    vid : { # vid
      "eid" : 0,
      "target neighbors" : "" [] #video id,
      "source neighbors" : "" [] #video id,
      "averageDailyView" : "",
      "averageDailyShare" : "",
      "averageWatchPercentage" : ""
    }
  }
'''

eid_to_vid = {}
vid_to_eid = {}

with open("./data/vevo_en_embeds_60k.txt", "r") as mapping_info:
  for map in mapping_info:
    eid, vid = map.split(",")[:2]
    eid = int(eid)
    output[vid] = {
      "eid" : eid,
      "target_neighbors" : [], #video id,
      "source_neighbors" : [] #video id,
    }
    eid_to_vid[eid] = vid
    vid_to_eid[vid] = eid
    
  
count = 0
for edge in network:
  source_eid = int(edge["Source"])
  target_eid = int(edge["Target"])
  source_vid = eid_to_vid[source_eid]
  target_vid = eid_to_vid[target_eid]
  output[source_vid]["target_neighbors"].append(target_vid)
  output[target_vid]["source_neighbors"].append(source_vid)
 
for video_info in videos_info:
  vid = video_info["id"]
  output[vid]["average_daily_view"] = video_info["insights"]["totalView"] / len(video_info["insights"]["dailyView"])
    
  if video_info["insights"].get("totalShare") == None or video_info["insights"].get("dailyShare") == None or video_info["insights"].get("avgWatch") == None:
    del output[vid]
    count += 1
  else:
    output[vid]["average_daily_share"] = video_info["insights"]["totalShare"] / len(video_info["insights"]["dailyShare"])
    if ((isodate.parse_duration(video_info["contentDetails"]["duration"]).total_seconds() / 60) != 0):
      output[vid]["average_watch_percentage"] = video_info["insights"]["avgWatch"] / (isodate.parse_duration(video_info["contentDetails"]["duration"]).total_seconds() / 60)
    else:
      print(isodate.parse_duration(video_info["contentDetails"]["duration"]).total_seconds())
    
print(count)
with open("output.json", "w") as output_file:
  output_file.write(json.dumps(output, indent=4))