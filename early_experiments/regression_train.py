from helper_functions import vectorize_list_items, pad_missing_cols, clean_col_names
import pickle
import pandas as pd
import numpy as np
import mysql.connector
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from catboost import CatBoostClassifier, Pool
from gensim.models import Word2Vec

# params
db_name = "adult"
table_name = "adult"
categorical_cols = ["workclass","education","maritalstatus","occupation","relationship","race","gender","nativecountry","income"]
dataset_path = "data/ds-generator/adult/talos_preprocessed"
w2v_size = len(categorical_cols)  #size of the word vectors generated later by word2vec

# load dataset pickle file created previously
with open(dataset_path, "rb") as fp:
   dataset = pickle.load(fp)

# connect to MySQL DB
# Might need to run "sudo service mysql start" beforehand if using WSL
cnx = mysql.connector.connect(user='python_user', password='pthon_user_password',
                              host='localhost',
                              database=db_name)
cursor = cnx.cursor()
full_table = pd.read_sql("SELECT * FROM " + table_name +";", cnx)
#close DB connection
cnx.close()

#  "QBE examples" from the dataset as input
X = [x[1] for x in dataset]
Y = [x[4] for x in dataset]

#split data into train and test sets
seed = 7

test_size = 0.0625
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# fit model no training data
model = XGBRegressor(tree_method="hist", n_estimators=64)
# model = CatBoostClassifier(iterations=2,
#                            depth=2,
#                            learning_rate=1,
#                            loss_function='Logloss',
#                            verbose=True)

print(X_train)
print("#######")
print(y_train)

# first training before loop to feed the model as parameter within the loop
model.fit(X_train[0], y_train[0])

for X, Y in zip(X_train[1:], y_train[1:]):
   model.fit(X, Y, xgb_model = model)

y_pred = []
# make predictions for test data
for entry in X_test:
   y_pred.append(model.predict(entry))

print(y_pred)
print("-------------------")
print(y_test)

# evaluate predictions
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))