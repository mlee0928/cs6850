from sklearn.neighbors import KNeighborsRegressor

X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(X, y)
neigh.predict([[1.5]])