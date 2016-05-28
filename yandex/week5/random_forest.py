import pandas
import numpy as np
import output_coursera as coursera
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score


data = pandas.read_csv('abalone.csv')
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
num_columns = len(data.columns)
X = data[data.columns[0:num_columns-1]]
y = data[data.columns[num_columns-1]]

regressor = RandomForestRegressor(random_state=1)
folder = KFold(n=X.shape[0], n_folds=5, random_state=1, shuffle=True)

scores = {}
for n_forest in range(1, 50 + 1):
    regressor.n_estimators = n_forest
    scores[n_forest] = np.mean(cross_val_score(regressor, X, y, scoring='r2', cv=folder, n_jobs=-1))


optimal = next(num for num, score in scores.items() if score >= 0.52)
coursera.output("size_of_forest.txt", str(optimal))
