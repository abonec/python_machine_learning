import pandas
import output_coursera as coursera
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack


def extract_features(data, text_vectorizer, dict_vectorizer, only_transform=None):
    descriptions = data['FullDescription'].str.lower().replace('[^a-z0-9]', ' ', regex=True)
    categories = data[['LocationNormalized', 'ContractTime']].fillna('nan').to_dict('records')

    if only_transform:
        descriptions = text_vectorizer.transform(descriptions)
        categories = dict_vectorizer.transform(categories)
    else:
        descriptions = text_vectorizer.fit_transform(descriptions)
        categories = dict_vectorizer.fit_transform(categories)

    return hstack([descriptions, categories])


data = pandas.read_csv('salary-train.csv')
text_vectorizer = TfidfVectorizer(min_df=5)
dict_vectorizer = DictVectorizer()
X = extract_features(data, text_vectorizer, dict_vectorizer)
y = data['SalaryNormalized']

model = Ridge(alpha=1, random_state=241)
model.fit(X, y)

data_test = pandas.read_csv('salary-test-mini.csv')
X_test = extract_features(data_test, text_vectorizer, dict_vectorizer, only_transform=True)
predictions = model.predict(X_test).tolist()
output = "{:.2f} {:.2f}".format(predictions[0], predictions[1])
coursera.output("salary_prediction.txt", output)

