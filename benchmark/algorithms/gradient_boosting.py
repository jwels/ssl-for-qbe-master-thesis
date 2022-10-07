import pickle, time
import xgboost as xgb
from sklearn.semi_supervised import SelfTrainingClassifier
import numpy as np
import pandas as pd

def run_gradient_boosting(examples = None, data = None):
    
    # load the one hot encoded databse
    # data = pd.read_csv("data/one_hot_encoded/onehot_"+ db_name + ".csv")

    # split example into training data and labels
    x_train = examples.loc[ : , examples.columns != 'label']
    y_train = np.ravel(examples.loc[ : , examples.columns == 'label'])
    
    # measure time
    start_time = time.time()
    
    # create model instance
    model = xgb.XGBClassifier()
    # train the model on initial QBE input example
    model = model.fit(X=x_train, y=y_train)
    # get the predictions for all the data from the newly trained model
    result = np.array(model.predict(data))
 
    # stop timing
    stop_time = time.time()

    return [result, stop_time-start_time]

def run_gradient_boosting_ssl(examples = None, data = None):
    
    # load the one hot encoded databse
    # data = pd.read_csv("data/one_hot_encoded/onehot_"+ db_name + ".csv")

    # split example into training data and labels
    x_train = examples.loc[ : , examples.columns != 'label']
    y_train = np.ravel(examples.loc[ : , examples.columns == 'label'])
    
    # measure time
    start_time = time.time()
    
    # create model instance
    base_model = xgb.XGBClassifier()
    model = SelfTrainingClassifier(base_model)
    # train the model on initial QBE input example
    model = model.fit(X=x_train, y=y_train)
    # get the predictions for all the data from the newly trained model
    result = np.array(model.predict(data))
 
    # stop timing
    stop_time = time.time()

    return [result, stop_time-start_time]