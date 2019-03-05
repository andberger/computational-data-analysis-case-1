import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


''' Exploratory data analysis '''
# use numpy array from now on
data = df.values

y = data[:100, 0]
X = data[:100, 1:]

# descriptive stats


# modelling


# estimating the root mean squared error
