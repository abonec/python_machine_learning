from sklearn.decomposition import PCA
import numpy as np
import pandas
import output_coursera as coursera


data = pandas.read_csv('close_prices.csv', index_col='date')

num_components = 10
decomposition = PCA(n_components=num_components)
reduced_data = decomposition.fit_transform(data)
ratio = 0
components_for_90_percent_dispersion = -1
for n in range(0, 10):
    ratio += decomposition.explained_variance_ratio_[n]
    if ratio >= 0.9:
        components_for_90_percent_dispersion = n+1
        break

coursera.output("components_for_90_percent_dispersion.txt", str(components_for_90_percent_dispersion))
###################################
first_component = reduced_data[:, 0]


djia_index = pandas.read_csv('djia_index.csv', index_col='date')['^DJI']

corr_coef = np.corrcoef(first_component, djia_index)

output = "{:.2f}".format(corr_coef[0][1])
coursera.output('correlation_coef.txt', output)
###################################
companies = data.columns.tolist()
companies_first_component = list(zip(decomposition.components_[0], companies))

valuable_company = max(companies_first_component)[1]
coursera.output('valuable_company.txt', valuable_company)


