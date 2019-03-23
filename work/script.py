import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, Lars

''' Utils '''
def get_most_common_value(column):
    return df[column].value_counts().head(1).index.values[0]

def replace_nan(x, replace_value):
    if pd.isnull(x):
        return replace_value
    else:
        return x

def plot_results(result_dict):
    pos = [1,2,3,4]
    values = [t[0] for t in result_dict.values()]
    labels = ['alpha = {0}'.format(round(t[1], 4)) for t in result_dict.values() if t[1] is not None]

    plt.bar(pos, values, align='center', alpha=0.5)
    plt.xticks([r + 1 for r in range(len(result_dict.keys()))], list(result_dict.keys()), rotation=90)
    plt.ylabel('R2')
    plt.title('Regression methods R2')
    for i in range(len(result_dict.keys()) - 1):
        if labels[i] is not None:
            plt.text(x = (pos[i+1])-0.25 , y = values[i+1]+0.01, s = labels[i], size = 6)
    plt.subplots_adjust(bottom= 0.20, top = 0.90)
    plt.show()

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
new_data = df.values[100:]
X = data[:, 1:]
y = data[:, 0]
X_new = new_data[:, 1:]

results = {}

# Linear regression with cross validation
lin_reg = LinearRegression(normalize=True)
R2s = cross_val_score(lin_reg, X, y, scoring='r2', cv=5)
R2 = np.mean(R2s)
results['Linear'] = (R2, None)
print(R2)

# Number of folds for cross validation
K = 10

# Hyperparameter values to try
params = {'alpha': list(np.logspace(-4, 3, 100))}

# Ridge regression with cross validation
ridge = Ridge(normalize=True)
ridge_regressor = GridSearchCV(ridge, params, scoring='r2', cv=K)
ridge_regressor.fit(X, y)
results['Ridge'] = (ridge_regressor.best_score_, ridge_regressor.best_params_['alpha'])
print(ridge_regressor.best_score_)

# Lasso regression with cross validation
lasso = Lasso(normalize=True)
lasso_regressor = GridSearchCV(lasso, params, scoring='r2', cv=K)
lasso_regressor.fit(X, y)
results['Lasso'] = (lasso_regressor.best_score_, lasso_regressor.best_params_['alpha'])
print(lasso_regressor.best_score_)

# ElasticNet regression with cross validation
elastic = ElasticNet(normalize=True)
elastic_regressor = GridSearchCV(elastic, params, scoring='r2', cv=K)
elastic_regressor.fit(X, y)
results['ElasticNet'] = (elastic_regressor.best_score_, elastic_regressor.best_params_['alpha'])
print(elastic_regressor.best_score_)

# Plot the results
plot = False
if plot:
    plot_results(results)

''' Lasso regression gives the best predictive performance.
    Let's finalize a Lasso regression model to make predictions on new data.
'''
lasso_final = lasso_regressor.best_estimator_
lasso_final.fit(X, y)
y_predict = lasso_final.predict(X_new)
import pdb; pdb.set_trace()
