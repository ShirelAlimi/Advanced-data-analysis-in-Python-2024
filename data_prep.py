import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, make_scorer
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns

fpath = "Data-set.csv"
cardf = pd.read_csv(fpath)
cardf.head()
cardf.info()

def data_prep(df):

    df = df[pd.to_datetime(df['Cre_date'], dayfirst=True, errors='coerce').notna()]
    df = df[pd.to_datetime(df['Repub_date'], dayfirst=True, errors='coerce').notna()]

    df['Cre_date'] = pd.to_datetime(df['Cre_date'], dayfirst=True)
    df['Repub_date'] = pd.to_datetime(df['Repub_date'], dayfirst=True)

    df['capacity_Engine'] = pd.to_numeric(df['capacity_Engine'], errors='coerce')
    df['Km'] = pd.to_numeric(df['Km'], errors='coerce')
    df['manufactor'] = df['manufactor'].replace('Lexsus', 'לקסוס')
    df['Gear'] = df['Gear'].replace('אוטומט', 'אוטומטית')
    df['Pic_num'] = df['Pic_num'].fillna(0).astype(int)

    categorical_columns = ['manufactor', 'model', 'Gear', 'Engine_type', 'Area', 'City']
    for col in categorical_columns:
        df[col] = df[col].astype('category')

    df['Current_Year'] = 2024
    df['age']=df['Current_Year']- df['Year']
    df['Km'] = df.groupby(['Year', 'Engine_type'])['Km'].transform(lambda x: x.fillna(x.mean()))
    df['capacity_Engine'] = df.groupby(['Engine_type', 'Year', 'model'])['capacity_Engine'].transform(lambda x: x.fillna(x.mean()))

    df.dropna(subset=['Km','Area','capacity_Engine','Gear'], inplace=True)
    columns_to_remove = ['Prev_ownership', 'Curr_ownership', 'Color', 'Supply_score','Test','Year','Current_Year']
    df.drop(columns=columns_to_remove, inplace=True)

    return df

X= cardf.drop(columns=['Price'])
y = cardf['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train = data_prep(X_train)
X_test = data_prep(X_test)
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
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

cv_scores = cross_val_score(elastic_net, X_train_top_scaled, y_train, cv=10, scoring=mse_scorer)
mean_cv_score = np.mean(cv_scores)

print(f'Mean CV Score (MSE): {-mean_cv_score}')
elastic_net.fit(X_train_top_scaled, y_train)

X_test_top_scaled = scaler.transform(X_test_top)
y_pred = elastic_net.predict(X_test_top_scaled)

print("\nFirst few predictions:")
y_pred[:5]