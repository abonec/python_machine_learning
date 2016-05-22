import pandas
import output_coursera as coursera
data = pandas.read_csv('titanic.csv', index_col='PassengerId')

males = data.loc[data.Sex == 'male']
females = data.loc[data.Sex == 'female']

output = "{0} {1}".format(len(males), len(females))

coursera.output('sub_1.txt', output)