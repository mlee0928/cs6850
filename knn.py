from sklearn.neighbors import KNeighborsRegressor
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

'''
Inputs:  1. aver_daily_view
         2. aver_daily_share
         3. aver_watch_time
         4. neighbor_engagement
Outputs: 1. aver_watch_percentage
         2. relative_engagement
'''
def knn(): 
    # Convert data to feature matrix 
    f = open('data/compile_data.json')
    data = json.load(f)
    X = []
    Y = []
    for vid in data: 
        x = []
        x.extend(data[vid]['neighbor_aver_daily_view'])
        x.extend(data[vid]['neighbor_aver_daily_share'])
        x.extend(data[vid]['neighbor_aver_watch_time'])
        x += data[vid]['neighbor_engagement']
        x += data[vid]['centrality']
        X += [x]
        y = []
        y += [data[vid]['aver_watch_percentage']]
        y += [data[vid]['relative_engagement']]
        Y += [y]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Run k-NN
    neigh = KNeighborsRegressor(n_neighbors=20)
    neigh.fit(X_train, y_train)

    # Test prediction
    test_point = []
    test_point += [data['--5u48IaR4M']['aver_daily_view']]
    test_point += [data['--5u48IaR4M']['aver_daily_share']]
    test_point += [data['--5u48IaR4M']['aver_watch_time']]
    test_point += data['--5u48IaR4M']['neighbor_engagement']

    test_gt = [data['--5u48IaR4M']["aver_watch_percentage"], data['--5u48IaR4M']['relative_engagement']]

    # test_pred = neigh.predict([test_point])
    pred = neigh.predict(X_test)
    r2 = r2_score(y_test, pred)

    count = len(y_test)
    losses = 0
    for i in range(count):
        losses += ((y_test[i][0] - pred[i][0])) ** 2
        losses += ((y_test[i][1] - pred[i][1])) ** 2
    losses /= count

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

    print(f"SMAPE: {smape(pred, y_test)}, MSE: {losses}")
    # print(f"Pred: {test_pred}, GT: {test_gt}")

    f.close()

knn()