import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from car_data_prep import prepare_data

fpath = "Data-set.csv"
cardf = pd.read_csv(fpath)

X = cardf.drop(columns=['Price'])
y = cardf['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train = prepare_data(X_train)
X_test = prepare_data(X_test)

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

col = ['Repub_date', 'Cre_date']
X_train.drop(columns=col, inplace=True)
X_test.drop(columns=col, inplace=True)

X_train, y_train = X_train.align(y_train, join='inner', axis=0)

model = GradientBoostingRegressor()
model.fit(X_train, y_train)

feature_importances = model.feature_importances_
important_indices = np.argsort(feature_importances)[-5:]
important_features = X_train.columns[important_indices]
important_scores = feature_importances[important_indices]

plt.figure(figsize=(10, 6))
plt.barh(important_features, important_scores, color=sns.color_palette('Set2'))
plt.xlabel('Feature Importance Score')
plt.title('Top 5 Important Features')
plt.show()

X_train_top = X_train[important_features]
X_test_top = X_test[important_features]

scaler = StandardScaler()
X_train_top_scaled = scaler.fit_transform(X_train_top)
elastic_net = ElasticNet()
elastic_net.fit(X_train_top_scaled, y_train)

joblib.dump(elastic_net, 'trained_model.pkl')
