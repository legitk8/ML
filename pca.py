import pandas as pd

df = pd.read_csv('./data/bank.csv')
df_nums = df.select_dtypes(['int64', 'float64'])
df_objs = df.select_dtypes(object)

# label encoding
from sklearn.preprocessing import LabelEncoder
for col in df_objs:
    le = LabelEncoder()
    df_objs[col] = le.fit_transform(df_objs[col])
    
# new dataframe (combined)
df_new = pd.concat([df_nums, df_objs], axis=1)

X = df_new.drop('y', axis=1)
y = df_new['y']
    
# split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# feature scaling
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# logistic regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

res = pd.DataFrame({'y Actual': y_test, 'y Predicted': y_pred})
# print(res)

# accuracy
from sklearn.metrics import accuracy_score, confusion_matrix
print('Accuracy: ', accuracy_score(y_test, y_pred)*100)
print(confusion_matrix(y_test, y_pred))
print('Shape: ', X.shape[1])
print(X_train)


from sklearn.decomposition import PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.20)

ss = StandardScaler()
X_train_pca = ss.fit_transform(X_train_pca)
X_test_pca = ss.transform(X_test_pca)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_pca, y_train)
y_pred = lr.predict(X_test_pca)

print('Accuracy: ', accuracy_score(y_test, y_pred)*100)
print(confusion_matrix(y_test, y_pred))
print('Shape: ', X_pca.shape[1])
print(X_pca)