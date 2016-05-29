import pandas
import math
import matplotlib.pyplot as plt
import numpy as np
import output_coursera as coursera
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble.forest import RandomForestClassifier

data = pandas.read_csv('gbm-data.csv')
X = data[data.columns[1:]].values
y = data[data.columns[0]].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

clf = GradientBoostingClassifier(n_estimators=250, random_state=241, verbose=True)


sigmoid = np.vectorize(lambda x: (1 / (1 + math.exp(-x))))


coursera.output('overfitting.txt', 'overfitting')

looses = {}
def plot_score(test_predictions, y_test, train_predictions, y_train, color, learning_rate):
    test_loss = [log_loss(y_test, pred) for pred in test_predictions]
    train_loss = [log_loss(y_train, pred) for pred in train_predictions]

    plt.plot(test_loss, color, linewidth=2)
    plt.plot(train_loss, color+'--', linewidth=2)
    looses[learning_rate] = test_loss

plt.figure()
colors = ['r', 'g', 'b', 'c', 'm']
learn_rates = [1, 0.5, 0.3, 0.2, 0.1]
for index, learning_rate in enumerate(learn_rates):
    clf.learning_rate = learning_rate
    clf.fit(X_train, y_train)
    test_predictions = clf.staged_predict_proba(X_test)
    train_predictions = clf.staged_predict_proba(X_train)
    plot_score(test_predictions, y_test, train_predictions, y_train, color=colors[index], learning_rate=learning_rate)

legends = [["Test {}".format(learn_rate), "Train {}".format(learn_rate)] for learn_rate in learn_rates]
legends = [item for sublist in legends for item in sublist]
plt.legend(legends)
plt.savefig("coursera_out/gradient_boosting.png")

min_loss_on_iteration = np.argmin(looses[0.2])
min_loss = looses[0.2][min_loss_on_iteration]
print("on iteration {} was loose {}".format(min_loss_on_iteration, min_loss))
coursera.output("min_loose_on_0.2.txt", "{:.2f} {}".format(min_loss, min_loss_on_iteration))


clf = RandomForestClassifier(n_estimators=36, random_state=241)
clf.fit(X_train, y_train)
predicts = clf.predict_proba(X_test)
loss = log_loss(y_test, predicts)
coursera.output("random_forest_loss_on_36_trees.txt", "{:.2f}".format(loss))
