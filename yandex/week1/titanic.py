import pandas
import output_coursera as coursera
data = pandas.read_csv('titanic.csv', index_col='PassengerId')

coursera.output('out1.txt', 'hello')