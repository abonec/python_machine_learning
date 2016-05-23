import sklearn.datasets
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor
import output_coursera as coursera

data = sklearn.datasets.load_boston()
X = data.data
y = data.target
X_scaled = scale(X)

parameters = np.linspace(start=1, stop=10, num=200)
best_p = 0
best_score = -100
for p in parameters:
    clf = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p)
    k_folder = KFold(X_scaled.shape[0], n_folds=5, random_state=42, shuffle=True)
    scores = cross_val_score(clf, X=X_scaled, y=y, scoring='mean_squared_error', cv=k_folder)
    mean = np.mean(scores)
    if mean > best_score:
        best_p = p
        best_score = mean

coursera.output('adjust_metric.txt', "{:.0f}".format(best_p))
