import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# load data
enc = OneHotEncoder(handle_unknown='ignore')
dataset = pd.read_csv('data/adult.csv', delimiter=",")
categorical_cols = ["workclass","education","marital-status","occupation","relationship","race","gender","native-country","income"]
#dataset.iloc[:,categorical_cols].astype("category")
encoded_dataset = pd.get_dummies(dataset, columns = categorical_cols)
encoded_dataset = encoded_dataset.drop('income_<=50K', 1)

# for col in categorical_cols:
#     dataset.iloc[:,col] = enc.fit_transform(dataset.iloc[:,col])    

print(encoded_dataset)
# split data into X and y
X = encoded_dataset.iloc[:,0:14]
Y = encoded_dataset["income_>50K"]
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
print(model)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
print("-----------------")
print(y_test, predictions)
print("-----------------")

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))