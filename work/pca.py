import pandas as pd
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('dataCase1.csv')

data = df.values
y = data[:100, 0]
X = data[:100, 1:96]

sklearn_pca = sklearnPCA(n_components=2)
X_std = StandardScaler().fit_transform(X)
Y_sklearn = sklearn_pca.fit_transform(X_std)

for name, col in zip(('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'), colors.values()):

    trace = dict(
        type='scatter',
        x=Y_sklearn[y==name,0],
        y=Y_sklearn[y==name,1],
        mode='markers',
        name=name,
        marker=dict(
            color=col,
            size=12,
            line=dict(
                color='rgba(217, 217, 217, 0.14)',
                width=0.5),
            opacity=0.8)
    )
    data.append(trace)

layout = dict(
        xaxis=dict(title='PC1', showline=False),
        yaxis=dict(title='PC2', showline=False)
)
fig = dict(data=data, layout=layout)
py.iplot(fig, filename='pca-scikitlearn')
