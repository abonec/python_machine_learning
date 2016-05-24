from sklearn.svm import SVC
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
import output_coursera as coursera
import numpy as np
import sklearn.datasets

newsgroup = sklearn.datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(newsgroup.data)
y = newsgroup.target

C = np.power(10.0, np.arange(-5, 6))
grid = {'C': C}
k_folder = KFold(X.shape[0], n_folds=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
grid_search = GridSearchCV(clf, grid, scoring='accuracy', cv=k_folder)
grid_search.fit(X, y)

optimal_parameters = {}
max_score = max(x.mean_validation_score for x in grid_search.grid_scores_)
optimal_c = next(x.parameters['C'] for x in grid_search.grid_scores_ if x.mean_validation_score == max_score)

clf.C = optimal_c
clf.fit(X, y)

feature_mappings = vectorizer.get_feature_names()
result = {
    'words': list(feature_mappings[i] for i in clf.coef_.indices),
    'values': list(abs(weight) for weight in clf.coef_.data),
}
coef = DataFrame(data=result)
coef = coef.sort_values(by='values', ascending=False)

words = coef.head(10)['words'].values.tolist()

output = " ".join(sorted(words))
coursera.output("svm_text_analyze.txt", output)
