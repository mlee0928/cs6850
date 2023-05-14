from sklearn.metrics import mean_squared_error, r2_score
import json
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

with open('data/compile_data.json', 'r', encoding="utf-8") as f:
    compile_data = json.load(f)
    f.close()

def sq_loss_fn(gt, pred, output1_w = 0.5, output2_w = 0.5):
    count = len(gt)
    losses = 0
    for i in range(count):
        losses += ((gt[i][0] - pred[i][0]) * output1_w) ** 2
        losses += ((gt[i][1] - pred[i][1]) * output2_w) ** 2
    losses /= count
    return losses

def abs_loss_fn(gt, pred, output1_w = 0.5, output2_w = 0.5):
    count = len(gt)
    losses = 0
    for i in range(count):
        losses += abs((gt[i][0] - pred[i][0]) * output1_w)
        losses += abs((gt[i][1] - pred[i][1]) * output2_w)
    losses /= count
    return losses

def smape(y_pred, y_test):
    # print(y_pred[0])
    n = len(y_pred)
    # print(n)
    acc = 0
    for i in range(n):
        a = abs(y_pred[i][0] - y_test[i][0])
        b = abs(y_pred[i][1] - y_test[i][1])
        c = y_test[i][0] + y_test[i][1]
        d = y_pred[i][0] + y_pred[i][1]
        acc += abs((a + b) / ((c + d) / 2))
    return acc / n

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
      # x.extend(compile_data[vid]["neighbor_aver_daily_view"])
      # x.extend(compile_data[vid]["neighbor_aver_daily_share"])
      # x.extend(compile_data[vid]['neighbor_aver_watch_percentage'])
      # x.extend(compile_data[vid]['neighbor_engagement'])
      # x.extend(compile_data[vid]['neighbor_centrality'])
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

    # The coefficients
    # print("Coefficients: \n", regr.coef_)
    # # The mean squared error
    # print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # # The coefficient of determination: 1 is perfect prediction
    # # print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))
    # print("SMAPE:", smape(y_pred, y_test))