import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

''' Utils '''
def get_most_common_value(column):
    return df[column].value_counts().head(1).index.values[0]

def replace_nan(x, replace_value):
    if pd.isnull(x):
        return replace_value
    else:
        return x

#A helper method for pretty-printing linear models
def pretty_print_linear(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)
                                   for coef, name in lst)

''' Read in data '''
df = pd.read_csv('dataCase1.csv')

''' Data preparation '''
# feature columns holding numerical data
columns_num = df.columns[1:96]

# feature columns holding categorical data
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

# drop original categorical columns
df = df.drop(columns=list(columns_cat))

# append the one-hot encodings of the categorical columns
for _, value in one_hot_encodings.items():
    df = pd.concat([df, value], axis=1)

''' Regression '''
# use numpy array from now on
data = df.values[:100]
X = data[:, 1:]
y = data[:, 0]

# Ridge regression with cross validation
from sklearn.linear_model import RidgeCV
ridge = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1], normalize=True, store_cv_values=True).fit(X, y)
score = ridge.score(X, y)
print(ridge.get_params())
print(score)
print(ridge.predict(df.values[100:105, 1:]))
print(ridge.alpha_)

# Lasso regression with cross validation
from sklearn.linear_model import LassoCV
lasso = LassoCV(cv=5, normalize=True).fit(X, y)
score_lasso = lasso.score(X, y)
print(score_lasso)
print(lasso.predict(df.values[100:105, 1:]))
print(lasso.alpha_)
