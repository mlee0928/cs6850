from sklearn.neighbors import KNeighborsRegressor
import json 

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
        x += [data[vid]['aver_daily_view']]
        x += [data[vid]['aver_daily_share']]
        x += [data[vid]['aver_watch_time']]
        x += data[vid]['neighbor_engagement']
        X += [x]
        y = []
        y += [data[vid]['aver_watch_percentage']]
        y += [data[vid]['relative_engagement']]
        Y += [y]
    
    # Run k-NN
    neigh = KNeighborsRegressor(n_neighbors=20)
    neigh.fit(X, Y)

    # Test prediction
    test_point = []
    test_point += [data['--5u48IaR4M']['aver_daily_view']]
    test_point += [data['--5u48IaR4M']['aver_daily_share']]
    test_point += [data['--5u48IaR4M']['aver_watch_time']]
    test_point += data['--5u48IaR4M']['neighbor_engagement']

    test_pred = neigh.predict([test_point])
    print(test_pred)

    f.close()

knn()