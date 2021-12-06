import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data = pd.read_csv('C:\\Users\\user\\Downloads\\assignments\\Module 8\\diabetes.csv')

data.info()

data.describe()

data.head()

data.shape

data.size


a = data.iloc[: , : -1].values
a
b = data.iloc[:, -1].values
b


x_train, x_test, y_train, y_test = train_test_split(a, b, test_size=.3, random_state=0)


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


DecisionTree_Classifier = DecisionTreeClassifier()
DecisionTree_Classifier.fit(x_train, y_train)


DecisionTreePredictions = DecisionTree_Classifier.predict(x_test)


accuracy_score(y_test, DecisionTreePredictions) # 0.7359307359307359



RandomForest_Classifier = RandomForestClassifier()
RandomForest_Classifier.fit(x_train, y_train)


RandomForestPredictions = RandomForest_Classifier.predict(x_test)


accuracy_score(y_test, RandomForestPredictions) # 0.7748917748917749

