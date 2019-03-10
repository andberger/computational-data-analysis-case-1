import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

''' Utils '''
def get_most_common_value(column):
    return df[column].value_counts().head(1).index.values[0]

def replace_nan(x, replace_value):
    if pd.isnull(x):
        return replace_value
    else:
        return x

''' Read in data '''
df = pd.read_csv('dataCase1.csv')

''' Data preparation '''
# Feature columns holding numerical data
columns_num = df.columns[1:96]

# Feature columns holding categorical data
columns_cat = df.columns[96:]

# Numerical nan's: replace missing values with the mean.
for i in range(len(columns_num)):
    mean = df[columns_num[i]].mean()
    df[columns_num[i]] = df[columns_num[i]].map(lambda x: replace_nan(x, mean))

# Categorical nan's: replace the missing entries with the most frequent one.
for i in range(len(columns_cat)):
    most_common_value = get_most_common_value(columns_cat[i])
    df[columns_cat[i]] = df[columns_cat[i]].map(lambda x: replace_nan(x, most_common_value))

# Use one-hot encoding to transform categorical data
one_hot_encodings = {}
for i in range(len(columns_cat)):
    df_dummies = pd.get_dummies(df[columns_cat[i]], prefix = columns_cat[i])
    one_hot_encodings[columns_cat[i]] = df_dummies

# Drop original categorical columns
df = df.drop(columns=list(columns_cat))

# Append the one-hot encodings of the categorical columns
for _, value in one_hot_encodings.items():
    df = pd.concat([df, value], axis=1)

''' Standardize '''
scaler = StandardScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

''' Regression '''
# Use numpy array from now on
data = df.values[:100]
X = data[:, 1:]
y = data[:, 0]

# Linear regression with cross validation
lin_reg = LinearRegression()
MSEs = cross_val_score(lin_reg, X, y, scoring='neg_mean_squared_error', cv=5)
mean_MSE = np.mean(MSEs)
print(mean_MSE)

# Number of folds for cross validation
K = 5

# Hyperparameter values to try
#params = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10]}
params = {'alpha': list(np.logspace(-4, 3, 100))}

# Ridge regression with cross validation
ridge = Ridge()
ridge_regressor = GridSearchCV(ridge, params, scoring='neg_mean_squared_error', cv=K)
ridge_regressor.fit(X, y)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

# Lasso regression with cross validation
lasso = Lasso()
lasso_regressor = GridSearchCV(lasso, params, scoring='neg_mean_squared_error', cv=K)
lasso_regressor.fit(X, y)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

# ElasticNet regression with cross validation
elastic = ElasticNet()
elastic_regressor = GridSearchCV(elastic, params, scoring='neg_mean_squared_error', cv=K)
elastic_regressor.fit(X, y)
print(elastic_regressor.best_params_)
print(elastic_regressor.best_score_)
