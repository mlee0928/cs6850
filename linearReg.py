from sklearn.metrics import mean_squared_error, r2_score
import json
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

with open('data/compile_data.json', 'r', encoding="utf-8") as f:
    compile_data = json.load(f)
    f.close()

Xs = []
Ys = []
for vid in compile_data.keys():
  x = [compile_data[vid]["aver_daily_view"], compile_data[vid]["aver_daily_share"], compile_data[vid]['aver_watch_time']]
  for i in compile_data[vid]['neighbor_engagement']:
    x.append(i)
  Xs.append(x)
  Ys.append([compile_data[vid]["aver_watch_percentage"], compile_data[vid]["relative_engagement"]])
  
X_train, X_test, y_train, y_test = train_test_split(Xs, Ys, test_size = 0.25)
regr = LinearRegression()
 
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))