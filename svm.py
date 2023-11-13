import pandas as pd

df = pd.read_csv('./data/Social_Network_Ads.csv')
X = df.drop(['Purchased'], axis=1)
y = df['Purchased']

# split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# linear SVM
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# print
print(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}))

# check accuracy
from sklearn.metrics import accuracy_score, confusion_matrix
print('Accuracy: ', accuracy_score(y_test, y_pred)*100, '%')
print(confusion_matrix(y_test, y_pred))