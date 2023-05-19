from sklearn.neighbors import KNeighborsRegressor
import json
from sklearn.model_selection import train_test_split
from utils import *

"""
Inputs:  1. neighbor_aver_daily_view
         2. neighbor_aver_daily_share
         3. neighbor_aver_watch_percentage
         4. neighbor_engagement
         5. network centrality
Outputs: 1. aver_watch_percentage
         2. relative_engagement
"""
lst_dict = {0: ["neighbor_aver_daily_view", "neighbor_aver_daily_share", "neighbor_aver_watch_percentage", "neighbor_engagement",
                        "neighbor_centrality"],
            1: ["neighbor_aver_daily_view", "neighbor_aver_daily_share",
                        "neighbor_aver_watch_percentage",
                        "neighbor_centrality"],
            2: ["neighbor_aver_daily_view", "neighbor_aver_daily_share", "neighbor_engagement", "neighbor_centrality"],
            3: ["neighbor_aver_daily_view", "neighbor_aver_daily_share", "neighbor_aver_watch_percentage", "neighbor_engagement"],
            4: ["neighbor_aver_daily_share", "neighbor_aver_watch_percentage", "neighbor_engagement", "neighbor_centrality"],
            5: ["neighbor_aver_daily_view", "neighbor_aver_watch_percentage", "neighbor_engagement", "neighbor_centrality"],
            6: ["neighbor_engagement"],
            7: ["neighbor_aver_watch_percentage"],
            8: ["neighbor_centrality"],
            9: ["neighbor_aver_daily_view"],
            10: ["neighbor_aver_daily_share"]
            }

for num in range(22):
    if num < 11:
        met = "l2"
    else:
        loss_fn = "l1"
    def knn():
        # Convert data to feature matrix
        f = open('data/compile_data.json')
        data = json.load(f)
        X = []
        Y = []
        for vid in data:
            all_keys = lst_dict[num % 11]

            lst = []
            for k in all_keys:
                lst.extend(data[vid][k])

            X += [lst]
            y = []
            y += [data[vid]['aver_watch_percentage']]
            y += [data[vid]['relative_engagement']]
            Y += [y]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Run k-NN
        neigh = KNeighborsRegressor(n_neighbors=5, metric=met)
        neigh.fit(X_train, y_train)

        # Test prediction
        test_point = []
        test_point += [data['--5u48IaR4M']['neighbor_aver_daily_view']]
        test_point += [data['--5u48IaR4M']['neighbor_aver_daily_share']]
        test_point += [data['--5u48IaR4M']['neighbor_aver_watch_percentage']]
        test_point += data['--5u48IaR4M']['neighbor_engagement']
        test_point += data['--5u48IaR4M']['neighbor_centrality']

        test_gt = [data['--5u48IaR4M']["aver_watch_percentage"], data['--5u48IaR4M']['relative_engagement']]

        # test_pred = neigh.predict([test_point])
        pred = neigh.predict(X_test)

        mse = sq_loss_fn(y_test, pred)
        mae = abs_loss_fn(y_test, pred)

        print(f"Number: {num}\nSMAPE: {round(smape(pred, y_test), 4)}, MSE: {round(mse, 4)}, MAE: {round(mae, 4)}")

        f.close()

    knn()