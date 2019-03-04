import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read data as csv
df = pd.read_csv('dataCase1.csv')

# data preparation
def get_most_common_value(column):
    return df[column].value_counts().head(1).index.values[0]

def replace_nan(x, replace_value):
    if pd.isnull(x):
        return replace_value
    else:
        return x

# Numerical nan's: replace missing values with the mean.
columns = df.columns[1:]
for i in range(len(columns) - 5):
    mean = df[columns[i]].mean()
    df[columns[i]] = df[columns[i]].map(lambda x: replace_nan(x, mean))

# Categorical nan's: replace the missing entries with the most frequent one.
#TODO: put in a loop
x96_mcv = get_most_common_value('X96') #B
x97_mcv = get_most_common_value('X97') #B
x98_mcv = get_most_common_value('X98') #D
x99_mcv = get_most_common_value('X99') #D
x100_mcv = get_most_common_value('X100') #D

df['X96'] = df['X96'].map(lambda x: replace_nan(x, x96_mcv))
df['X97'] = df['X97'].map(lambda x: replace_nan(x, x97_mcv))
df['X98'] = df['X98'].map(lambda x: replace_nan(x, x98_mcv))
df['X99'] = df['X99'].map(lambda x: replace_nan(x, x98_mcv))
df['X100'] = df['X100'].map(lambda x: replace_nan(x, x100_mcv))



# Exploratory data analysis
# use numpy array from now on
data = df.values

y = data[:100, 0]
X = data[:100, 1:]

# descriptive stats


# modelling


# estimating the root mean squared error
