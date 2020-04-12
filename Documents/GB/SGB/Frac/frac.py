import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

train_data = pd.read_csv("/Users/ramilguliev/Documents/GB/SGB/Frac/train.csv", engine='python')
test_data = pd.read_csv("/Users/ramilguliev/Documents/GB/SGB/Frac/test.csv", engine='python')

train_data = train_data[0:800]
test_data = test_data[801:]

drop_columns = ["proppant"]
test_data.drop(labels=drop_columns, axis=1, inplace=True)

y_train = train_data["proppant"]
train_data.drop(labels="proppant", axis=1, inplace=True)

full_data = train_data.append(test_data)

full_data = full_data.fillna(full_data.mean())

X_train = full_data.values[0:800]
X_test = full_data.values[801:]

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

state = 12  
test_size = 0.30  
  
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,  
    test_size=test_size, random_state=state)

lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0)
    gb_clf.fit(X_train, y_train)

    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_val, y_val)))
