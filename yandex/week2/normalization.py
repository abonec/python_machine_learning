from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import pandas
import output_coursera as coursera


data_train = pandas.read_csv('perceptron-train.csv', header=None)
data_test = pandas.read_csv('perceptron-test.csv', header=None)

y_train = data_train[data_train.columns[0]]
y_test = data_test[data_test.columns[0]]

X_train = data_train[data_train.columns[1:]]
X_test = data_test[data_test.columns[1:]]

clf = Perceptron(random_state=241)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
score_without_normalization = accuracy_score(y_true=y_test, y_pred=predictions)
print(score_without_normalization)

scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

clf.fit(X_train_normalized, y_train)
predictions = clf.predict(X_test_normalized)
score_with_normalization = accuracy_score(y_true=y_test, y_pred=predictions)
print(score_with_normalization)

result = score_with_normalization - score_without_normalization
output = "{:.3f}".format(result)
print(output)
coursera.output('normalization.txt', output)



