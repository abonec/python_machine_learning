from sklearn.tree import DecisionTreeClassifier
import pandas
import output_coursera as coursera

data = pandas.read_csv('titanic.csv', index_col='PassengerId')
fields = ['Pclass', 'Fare', 'Age', 'Sex']
all_fields = fields + ['Survived']
data = data[all_fields].dropna()
X = data[fields]
y = data['Survived']

X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})

clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)

importances = list(zip(clf.feature_importances_, fields))
importances = sorted(importances, key=lambda item: item[0], reverse=True)
importances = list(map(lambda item: item[1], importances[0:2]))

output = " ".join(importances)
coursera.output('decision_tree_importancies.txt', output)
