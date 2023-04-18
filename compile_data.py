# import pickle
# import isodate
#
#
# # Open the file in binary mode
# with open('dataverse_files/engagement_map.p', 'rb') as f:
#     # Load the pickled data
#     data = pickle.load(f)
#
# # Use the data
# print(type(data), data.keys(), len(data["duration"]))
#
#
# duration = isodate.parse_duration("PT2M44S")
# total_seconds = duration.total_seconds()
#
# print(total_seconds)

import json

# total watch time, total view time, and duration --> average watch percentage
# output: dictionary with id, duration, and average watch percentage
output = []
filename = "vevo_en_videos_60k"
with open(f'dataverse_files/{filename}.json', 'r', encoding="utf-8") as f:
    data = json.load(f)

    for dictionary in data:
        vid_id = dictionary['id']
        duration = dictionary['contentDetails']['duration']
        total_view = dictionary['insights']['totalView']
        total_watch = dictionary['insights']['avgWatch'] * len(dictionary['insights']['dailyWatch'])

        aver_watch_time = total_watch / total_view
        aver_watch_percentage = aver_watch_time / duration

        new_d = {"id": vid_id, "duration": duration, "average watch percentage": aver_watch_percentage}

        output.append(new_d)

with open(f'dataverse_files/watch_per_{filename}.json', 'w') as f:
    json.dump(output, f)
