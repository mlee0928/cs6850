import json
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from utils import *

with open('data/compile_data.json', 'r', encoding="utf-8") as f:
    compile_data = json.load(f)
    f.close()

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


for num in range(11):
    Xs = []
    Ys = []
    for vid in compile_data.keys():
      x = []
      all_keys = lst_dict[num % 11]
      for k in all_keys:
          x.extend(compile_data[vid][k])

      Xs.append(x)
      Ys.append([compile_data[vid]["aver_watch_percentage"], compile_data[vid]["relative_engagement"]])

    X_train, X_test, y_train, y_test = train_test_split(Xs, Ys, test_size=0.2, random_state=42)
    mag = np.linalg.norm(X_train)
    X_train = X_train / mag
    mag = np.linalg.norm(X_test)
    X_test = X_test / mag
    regr = LinearRegression()

    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)

    mse = sq_loss_fn(y_test, y_pred)
    mae = abs_loss_fn(y_test, y_pred)

    print(f"Number: {num} - SMAPE: {round(smape(y_pred, y_test), 4)}, MSE: {round(mse, 4)}, MAE: {round(mae, 4)}")