import pandas as pd

# get data
data = pd.read_csv('./data/buy.csv')

X = data.iloc[:,0:-1].values
print('Features:\n', X)
y = data.iloc[:,-1].values
print('\nTarget:', y)

# initialize S and G
feature_length = len(X[0])
S = ['phi']*feature_length
G = [['?' for i in range(feature_length)] for j in range(feature_length)]

for i, features in enumerate(X):
    if y[i] == 'yes':
        for j, feature in enumerate(features):
            if S[j] == 'phi':
                S[j] = feature
                G[j][j] = '?'
            elif S[j] != feature:
                S[j] = '?'
                G[j][j] = '?'
    else:
        for j, feature in enumerate(features):
            if S[j] == 'phi':
                G[j][j] = '?'
            elif feature != S[j]:
                G[j][j] = S[j]
            else:
                G[j][j] = '?'
            
G = [g for g in G if g != ['?', '?', '?', '?', '?']]

print('Most Specific : ', S)
print('Most Generic : ', G)
