import pandas
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
import output_coursera as coursera


def get_scores(folder, X_eval, y_eval):
    means = {}
    result_k = 0
    result_score = 0
    for k in range(1, 51):
        scores = []
        for train_index, test_index in folder:
            X_train, X_test = X_eval.iloc[train_index], X_eval.iloc[test_index]
            y_train, y_test = y_eval.iloc[train_index], y_eval.iloc[test_index]
            clf = KNeighborsClassifier(n_neighbors=k)
            clf.fit(X_train, y_train)
            scores.append(clf.score(X_test, y_test))
        means[k] = np.mean(scores)
        current_score = np.mean(scores)
        if current_score > result_score:
            result_score = current_score
            result_k = k

    return result_k, result_score

data = pandas.read_csv('wine.data')
X = data[data.columns[1:]]
y = data[data.columns[0]]
k_folder = KFold(X.shape[0], n_folds=5, random_state=42, shuffle=True)
####################################
best_k, best_score = get_scores(folder=k_folder, X_eval=X, y_eval=y)
coursera.output('sub1.txt', str(best_k))
coursera.output('sub2.txt', "{:.2f}".format(best_score))
####################################
X_scaled = pandas.DataFrame(scale(X))
k_folder = KFold(X.shape[0], n_folds=5, random_state=42, shuffle=True)
best_k, best_score = get_scores(folder=k_folder, X_eval=X_scaled, y_eval=y)
coursera.output('sub3.txt', str(best_k))
coursera.output('sub4.txt', "{:.2f}".format(best_score))


