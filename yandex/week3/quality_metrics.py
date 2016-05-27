import pandas
import output_coursera as coursera


data = pandas.read_csv('classification.csv')

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
