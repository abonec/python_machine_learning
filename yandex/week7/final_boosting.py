import pandas
import numpy as np
import datetime
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

features_set = pandas.read_csv('features.csv')
X = features_set.drop(['duration', 'radiant_win', 'tower_status_radiant', 'tower_status_dire', 'barracks_status_radiant', 'barracks_status_dire'], axis=1)
y = features_set[['radiant_win']].values.ravel()

matches_count = len(X)

# Все признаки, которые имеют пропуски, имеют суффикс _time, что говорит о том, что они просто не успели наступить за
# первые пять минут. Все остальные признаки с пропусками имеют такие же префиксы, что означает, что они с ними связаны и
# несут уточняющую информацию
features_data_count = X.count()
missing = features_data_count[features_data_count < matches_count]
missing = missing.apply(lambda x: "missing {} of {}".format(matches_count - x, matches_count))
print(missing)


X = X.fillna(0)


size = 0
score = 0
for forest_size in [10, 20, 30, 50, 150, 300]:
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


