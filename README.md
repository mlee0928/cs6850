# cs6850
### `vevo_en_videos_60k.json` Format:
```
 {
     "id": "uJz3sRJ1fTQ",
     "snippet": {"publishedAt": "2018-07-12T23:01:00.000Z",
                 "channelId": "UCKonA-DOxJbL0VHUDlc3vsA",
                 "title": "Bury Tomorrow - Adrenaline (Audio)",
                 "description": "collapsed long text description...",
                 "thumbnails": "https://i.ytimg.com/vi/uJz3sRJ1fTQ/default.jpg",
                 "channelTitle": "BuryTomorrowVEVO",
                 "tags": ["tag 1", "tag 2", "tag n"],
                 "categoryId": "10"
                 },
     "contentDetails": {"duration": "PT2M44S",
                        "definition": "hd",
                        "caption": "false",
                        "licensedContent": true,
                        "regionRestriction": {"allowed": ["country 1", "country 2", "country n"]}
                        },
     "insights": {"dailyView": [1163, 3407, 1100, 850, 1131, 1156, 1197, 969, 929, 666, 529, 626, 697, 693, 721, 700, 542, 524, 614, 615, 572, 494, 515, 425, 343, 489, 447, 403, 435, 424, 347, 284, 321, 298, 342, 365, 358, 296, 199, 325, 288, 356, 355, 335, 273, 225, 286, 276, 321, 322, 305, 263, 236, 265, 263, 309, 287, 310, 213, 209, 253, 284, 230, 241, 221, 193, 180, 222, 241, 242, 226, 237, 200, 166, 219, 242, 179, 226, 299, 227, 226, 299, 266, 196, 207, 185, 143, 140, 187, 196, 168, 182, 208, 148, 150, 215, 197, 204, 189, 193, 179, 148, 174, 169, 172, 197, 209, 147, 102, 162, 132, 191, 167, 151],
                  "startDate": "2018-07-12",
                  "totalView": 42935,
                  "dailyShare": [38, 77, 28, 12, 19, 19, 10, 10, 7, 9, 6, 1, 6, 1, 7, 4, 4, 3, 7, 3, 3, 5, 1, 3, 4, 1, 6, 1, 2, 3, 2, 2, 0, 3, 4, 0, 2, 0, 0, 3, 2, 2, 0, 1, 0, 4, 1, 2, 2, 7, 1, 2, 1, 4, 1, 8, 2, 3, 0, 4, 4, 0, 0, 1, 0, 0, 2, 0, 4, 5, 3, 2, 1, 3, 2, 5, 0, 2, 1, 1, 0, 0, 0, 1, 5, 4, 2, 2, 1, 0, 2, 2, 1, 1, 2, 2, 1, 1, 0, 2, 2, 0, 1, 1, 0, 0, 4, 0, 1, 1, 1, 3, 1, 1],
                  "totalShare": 444,
                  "dailyWatch": [1898.31666667, 6293.01666667, 2077.51666667, 1608.01666667, 2222.45, 2398.23333333, 2487.4, 2170.83333333, 1885.93333333, 1338.36666667, 1048.16666667, 1320.76666667, 1578.21666667, 1585.98333333, 1631.46666667, 1567.1, 1214.91666667, 1172.81666667, 1390.43333333, 1385.86666667, 1231.33333333, 1110.85, 1134.91666667, 913.533333333, 723.916666667, 1130.81666667, 1010.56666667, 912.466666667, 1035.6, 946.15, 754.583333333, 608.316666667, 727.616666667, 656.533333333, 792.433333333, 823.966666667, 790.466666667, 644.35, 430.3, 711.7, 652.6, 805.55, 781.383333333, 716.933333333, 592.65, 447.166666667, 603.6, 607.616666667, 709.566666667, 718.266666667, 683.466666667, 592.65, 524.35, 591.1, 584.416666667, 669.016666667, 627.733333333, 694.783333333, 474.683333333, 451.616666667, 547.45, 605.266666667, 491.416666667, 483.933333333, 498.966666667, 424.4, 365.8, 481.5, 547.3, 510.45, 492.216666667, 505.716666667, 417.716666667, 366.766666667, 521.383333333, 499.616666667, 372.016666667, 498.55, 675.35, 473.1, 439.916666667, 632.05, 582.916666667, 422.15, 447.4, 404.933333333, 312.133333333, 277.3, 411.4, 426.55, 346.8, 418.716666667, 424.516666667, 312.883333333, 332.633333333, 478.633333333, 457.266666667, 453.95, 415.966666667, 423.9, 389.6, 314.6, 352.3, 364.666666667, 391.75, 452.816666667, 450.4, 314.033333333, 224.166666667, 369.966666667, 294.166666667, 402.133333333, 357.85, 305.416666667],
                  "avgWatch": 2.121350491053197,
                  "endDate": "2018-11-02"
                  },
     "topics": ["Music", "Rock_music"]
 }
 ```
Note: one dictionary for each video

### Units
- duration - seconds
- average watch percentage - percent

### `compile_data.json` Format
`compile_data.json` is the full set of data we have after preprocessing
```
{"vid_id1": 
    {
        "eid": 1, 
        "neighbors": ["EnIR91t4qgY", "RAC5Rv4cOhE", "d7cVLE4SaN0", "eCGV26aj-mM"], 
        "aver_daily_share": 332.3450508788159, 
        "duration": 242.0, 
        "total_view": 80160391, 
        "total_watch_time": 3131.3522672762774, 
        "aver_watch_time": 2.896718101088138, 
        "aver_watch_percentage": 1.1969909508628669, 
        "relative_engagement": 1, 
        "aver_daily_view": 74153.92321924145, 
        "source_neighbors": ["EnIR91t4qgY", "RAC5Rv4cOhE"], 
        "target_neighbors": ["d7cVLE4SaN0", "eCGV26aj-mM"], 
        "neighbor_aver_daily_view": [0.91986952, 0.33629022, 0.2016786, 0.0086403, 0.0], 
        "neighbor_aver_daily_share": [0.8843747, 0.364756, 0.2909698, 0.0133260, 0.0], 
        "neighbor_aver_watch_time": [0.50005693, 0.491883306, 0.551358489, 0.45179, 0.0], 
        "neighbor_aver_watch_percentage": [0.5187526, 0.47441278, 0.5215076, 0.483884738, 0.0], 
        "neighbor_centrality": [-2.343023e-18, -1.844017e-18, 6.27541e-19, -6.354093e-19, 0.0], 
        "neighbor_engagement": [0.5, 0.5, 0.5, 0.5, 0.0]
    }, 
 "vid_id2":
    {
        ...
    }
}
```

### File Descriptions
- `network.py` - parses the `vevo_en_videos_60k.json` to get relevant nodes and edges
- `compile_data.py` - further calculates and compiles the full dataset we need for our models
- `api_parser.py` - Youtube API parser to get video neighbors
- `knn.py`, `gnn.py`, `linearReg.py` - our models
- `utils.py` - functions for test loss calculation
- `examine_graphs.py` - a helper file for us to look at loss plots in more detail