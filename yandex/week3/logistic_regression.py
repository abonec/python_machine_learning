import pandas
import numpy as np
import math
from sklearn.metrics import roc_auc_score


class LogisticRegression:
    def __init__(self, k=0.1, stop_value=1e-5, n_iter=10000, c=0.1, regularization=True):
        self.k, self.stop_value, self.n_iter, self.c = k, stop_value, n_iter, c
        self.regularization = regularization
        self.w1 = self.w2 = 0.0

    def fit(self, X, y):
        self.l = X.shape[0]
        for x in range(0, self.n_iter):
            sum1 = 0.0
            sum2 = 0.0
            for index, row in X.iterrows():
                x1 = row.data[0]
                x2 = row.data[1]
                y_label = y[index]
                sub = 1 - (1 / (1 + math.exp(-y_label * (self.w1*x1 + self.w2*x2))))

                sum1 += y_label * x1 * sub
                sum2 += y_label * x2 * sub

            w1_gradient = self.k * (1 / self.l) * sum1
            w2_gradient = self.k * (1 / self.l) * sum2
            if self.regularization:
                w1_gradient -= (self.k * self.c * self.w1)
                w2_gradient -= (self.k * self.c * self.w2)
            self.w1 += w1_gradient
            self.w2 += w2_gradient
            if math.fabs(w1_gradient) <= self.stop_value and math.fabs(w2_gradient) <= self.stop_value:
                break
            print(x, self.w1, self.w2)

    def predict(self, x1, x2):
        return 1 / (1 + math.exp( -self.w1 * x1 - self.w2 * x2))


data = pandas.read_csv('data-logistic.csv', header=None)
X = data[data.columns[1:]]
y = data[data.columns[0]]

clf = LogisticRegression()
clf.fit(X, y)
# clf = LogisticRegression(regularization=None)
# clf.fit(X, y)

for index, row in X.iterrows():
    x1 = row.data[0]
    x2 = row.data[1]
    y_true = y[index]
    y_given = clf.predict(x1, x2)
    # print("was {} give {}".format(y_true, y_given))


print(X.iloc[:1])
