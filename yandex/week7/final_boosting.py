import pandas
import numpy as np
import datetime
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

features_set = pandas.read_csv('features.csv')
X = features_set.drop(['duration', 'radiant_win', 'tower_status_radiant', 'tower_status_dire', 'barracks_status_radiant', 'barracks_status_dire'], axis=1)
y = features_set[['radiant_win']].values.ravel()

matches_count = len(X)

features_data_count = X.count()
missing = features_data_count[features_data_count < matches_count]
missing = missing.apply(lambda x: "missing {} of {}".format(matches_count - x, matches_count))
print(missing)


X = X.fillna(0)


# ===================== GradientBoosting ==============================

size = 0
score = 0
for forest_size in [10, 20, 30, 50, 150, 300]:
    # break
    start_time = datetime.datetime.now()
    clf = GradientBoostingClassifier(n_estimators=forest_size)
    k_folder = KFold(X.shape[0], n_folds=5, shuffle=True)
    scores = cross_val_score(clf, X=X, y=y, cv=k_folder, scoring='roc_auc')
    current_score = np.mean(scores)
    print("for {} trees mean score has been {} and time elapsed {}".format(forest_size, current_score, datetime.datetime.now() - start_time))
    if score < current_score:
        score = current_score
        size = forest_size

print("best score was for {} forest size: {}".format(size, score))

# ===================LogisticRegression=================
features = X

X = StandardScaler().fit_transform(features)

def train_logistic(features, target, label):
    clf = LogisticRegression(penalty='l2')
    k_folder = KFold(features.shape[0], n_folds=5, shuffle=True)
    start_time = datetime.datetime.now()
    score = cross_val_score(clf, X=features, y=target, cv=k_folder, scoring='roc_auc').mean()
    print("{}: score {} given by logistic regression in {}".format(label, score, datetime.datetime.now() - start_time))

train_logistic(X, y, 'Scaled')

# drop category
category_features = ['lobby_type']
for i in range(1, 5+1):
    category_features.append("r{}_hero".format(i))
    category_features.append("d{}_hero".format(i))

X = features.drop(category_features, axis=1)
X = StandardScaler().fit_transform(X)

train_logistic(X, y, 'Drop categorial features')

