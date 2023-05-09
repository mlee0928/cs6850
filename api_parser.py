import googleapiclient.discovery
import isodate
import json

keys = []
with open("data/key.txt", 'r') as f:
    for line in f:
        key = line.strip()
        keys.append(key)

print(keys)

movie_data = {}

count = 1
key = keys[0]
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=key)
print(key)
# flag = True

# while True:
#     if count >= 40000:
#         break
#     if count % 10000 == 0:
#         print(count)
#         key = keys[int(count / 10000)]
#         print(key)
#     count += 1



with open("data/tweeted_videos/tweeted_videos/education.json", "r", encoding="utf-8") as videos:
    for line in videos:
        if count >= 40000:
            break
        if count % 10000 == 0:
            print(count)
            key = keys[int(count / 10000)]
            youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=key)
            print(key)
        v = json.loads(line)
        vid = v["id"]
        if v["insights"].get("totalShare") == None or v["insights"].get("dailyShare") == None or v["insights"].get("avgWatch") == None or (isodate.parse_duration(v["contentDetails"]["duration"]).total_seconds()) == 0:
            continue
        else:
            try:
                # print(v)
                average_daily_view = int(v["insights"]["totalView"]) / len(v["insights"]["dailyView"])
                average_daily_share = int(v["insights"]["totalShare"]) / len(v["insights"]["dailyShare"])
                duration = v['contentDetails']['duration']
                duration = isodate.parse_duration(duration)
                duration = duration.total_seconds()
                total_view = int(v['insights']['totalView'])
                total_watch_time = float(v['insights']['avgWatch']) * len(v['insights']['dailyWatch'])
                aver_watch_time = float(v['insights']['avgWatch'])
                aver_watch_percentage = aver_watch_time / duration * 100
            except:
                continue
        # Call the search.list method to get the related videos for a specific video
        try:
            # https://developers.google.com/youtube/v3/docs/search/list?apix_params=%7B%22part%22%3A%5B%22snippet%22%5D%2C%22maxResults%22%3A15%2C%22relatedToVideoId%22%3A%22Ks-_Mh1QhMc%22%2C%22type%22%3A%5B%22video%22%5D%7D&apix=true
            request = youtube.search().list(
                part="snippet",
                relatedToVideoId=vid,
                type="video",
                maxResults=20
            )
            response = request.execute()
            count += 1
            # print("good")
        except Exception as e:
            print(e)
            print("bad:", vid)
            continue

        # Loop through the response and print the video IDs of the related videos
        target_neighbors = []
        for item in response["items"]:
            target_neighbors.append(item["id"]["videoId"])

        movie_data[vid] = { "neighbors": target_neighbors, "aver_daily_share": average_daily_share,
                            "duration": duration, "total_view": total_view, "total_watch_time": total_watch_time,
                            "aver_watch_time": aver_watch_time, "aver_watch_percentage": aver_watch_percentage,
                            "relative_engagement": "", "aver_daily_view": average_daily_view,
                            "source_neighbors":  [],
                            "target_neighbors":  target_neighbors}

with open("data/education_data.json", "w", encoding="utf-8") as f:
    json.dump(movie_data, f)
