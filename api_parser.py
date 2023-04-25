import googleapiclient.discovery

with open("key.txt", 'r') as f:
    key = f.readline()
    key = key.strip()

# Build the resource for the YouTube API
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=key)

# Call the search.list method to get the related videos for a specific video
response = youtube.search().list(
    part="id",
    type="video",
    relatedToVideoId="VIDEO_ID",
    maxResults=10
).execute()

# Loop through the response and print the video IDs of the related videos
for item in response["items"]:
    print(item["id"]["videoId"])