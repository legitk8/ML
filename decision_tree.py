import pandas as pd

df = pd.read_csv('./data/play_tennis.csv')

df_num = df.select_dtypes(['int64', 'float64'])
df_objs = df.select_dtypes(object)

from sklearn.preprocessing import LabelEncoder
for col in df_objs:
    le = LabelEncoder()
    df_objs[col] = le.fit_transform(df_objs[col])
    
df_new = pd.concat([df_num, df_objs], axis=1)

X = df_new.drop(['Play Tennis'], axis=1)
y = df_new['Play Tennis']

# split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy', max_depth=4)

dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

# part to generate image
from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(dtc, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = X.columns.tolist(),class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('test.png')
Image(graph.create_png())

print(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}))