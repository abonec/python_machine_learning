import pandas
import re
import output_coursera as coursera
from scipy.stats.stats import pearsonr
data = pandas.read_csv('titanic.csv', index_col='PassengerId')
#############################
males = data.loc[data.Sex == 'male']
females = data.loc[data.Sex == 'female']

output = "{0} {1}".format(len(males), len(females))

coursera.output('sub_1.txt', output)
#############################
survived = data.loc[data.Survived == 1]
survived_fraction = len(survived) / len(data) * 100

output = "{:.2f}".format(survived_fraction)

coursera.output('sub_2.txt', output)
#############################
first_class_passengers = data.loc[data.Pclass == 1]
first_class_passengers_fraction = len(first_class_passengers) / len(data) * 100

output = "{:.2f}".format(first_class_passengers_fraction)
coursera.output('sub_3.txt', output)
#############################
ages = data['Age'].dropna()
ages_mean = ages.mean()
ages_median = ages.median()

output = "{:.2f} {:.0f}".format(ages_mean, ages_median)
coursera.output('sub_4.txt', output)
#############################
correlation = pearsonr(data['SibSp'], data['Parch'])[0]
output = "{:.2f}".format(correlation)
coursera.output('sub_5.txt', output)
#############################

# first_name explanation got here:
# https://www.kaggle.com/c/titanic/forums/t/2885/a-question-with-the-names/17477
def process_names(string):
    mrs_reg = r'(Mrs\.|Miss\.|Dr\.|Ms.|Mlle.|Lady.|Mme.)\s?'
    name = string.split(", ")[1]
    name = re.sub(mrs_reg, '', name)
    if '(' in name and ')' in name:
        name = re.search(r'\(\"?(.*)\"?\)', name).group(1)
        name = re.sub(r'(Mrs )', '', name)
    name = re.sub(r'\"', '', name)
    name = name.split(" ")[0]
    # name = [split[1, len(split) - 1]].join(" ")
    return name.lower()
names = females['Name'].apply(process_names)
name = names.value_counts().idxmax().capitalize()

coursera.output('sub_6.txt', name)
#############################
