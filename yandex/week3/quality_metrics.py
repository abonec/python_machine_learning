import pandas
import output_coursera as coursera
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve


data = pandas.read_csv('classification.csv')
y_true = data['true']
y_pred = data['pred']

#######################################################
TP = FP = FN = TN = 0

for _, row in data.iterrows():
    true, pred = row['true'], row['pred']
    if true == 1:
        if pred == 1:
            TP += 1
        else:
            FN += 1
    else:
        if pred == 1:
            FP += 1
        else:
            TN += 1

output = "{} {} {} {}".format(TP, FP, FN, TN)
coursera.output("quality_metrics_1.txt", output)
#######################################################
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
output = "{:.2f} {:.2f} {:.2f} {:.2f}".format(accuracy, precision, recall, f1)
coursera.output("quality_metrics_2.txt", output)
#######################################################
data = pandas.read_csv('scores.csv')
algorithms = ['logreg', 'svm', 'knn', 'tree']
y_true = data['true']
roc_aucs = {}
for algorithm in algorithms:
    column_name = "score_{}".format(algorithm)
    roc_aucs[column_name] = roc_auc_score(y_true, data[column_name])
best_alg = max(roc_aucs, key=roc_aucs.get)
coursera.output("quality_metrics_3.txt", best_alg)
#######################################################
pr_scores = {}
for algorithm in algorithms:
    column_name = "score_{}".format(algorithm)
    precision, recall, thresholds = precision_recall_curve(y_true, data[column_name])
    recall = recall.tolist()
    min_70_percent_recall = next(rec for rec in sorted(recall) if rec >= 0.7)
    index_of_recall = recall.index(min_70_percent_recall)
    pr_scores[column_name] = precision[index_of_recall]

best_alg = max(pr_scores, key=pr_scores.get)
coursera.output("quality_metrics_4.txt", best_alg)
#######################################################
