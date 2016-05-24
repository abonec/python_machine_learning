from sklearn.svm import SVC
import pandas
import output_coursera as coursera

data = pandas.read_csv('svm-data.csv', header=None)
X = data[data.columns[1:]]
y = data[data.columns[0]]

clf = SVC(C=100000, kernel='linear', random_state=241)
clf.fit(X,y)

result = clf.support_ + 1

output = " ".join(str(x) for x in sorted(result))
coursera.output("support_vectors.txt", output)
