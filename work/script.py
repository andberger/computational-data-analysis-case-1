import pandas as pd

# read data as csv
df = pd.read_csv('dataCase1.csv')

# data preparation

# Encode alphanumeric characters like so, A->1, B->2,..., Z -> 26. In case of "NA", do NA->0

for i in range(len(df.columns)):
    column = df.columns[i]
    if df[column].dtype == 'object':
        df[column] = df[column].apply(lambda x: ord(x) - 64 if type(x) == str and x is not 'NA'
                                else (0 if type(x) == str and x is 'NA' else x))

import pdb; pdb.set_trace()



# descriptive stats


# modelling


# estimating the root mean squared error
