import pandas as pd

dataframe = pd.read_csv('./data/enjoy_sport.csv')
 
X = dataframe.iloc[:,:-1].values
Y = dataframe.iloc[:,-1].values

h = ['phi']*len(X[0])

for i,features in enumerate(X):
    # only positive cases
    if Y[i] == 'yes':
        for j, feature in enumerate(features):
            if h[j] == 'phi':
                h[j] = feature
                
            elif h[j] != feature:
                h[j] = '?'
                
print('Final hypothesis : ', h)