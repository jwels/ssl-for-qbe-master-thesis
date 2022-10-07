import enum
import pickle
import sys
from typing import Mapping
from sqlgenerator.models import Adult
from sklearn import tree
from sklearn.semi_supervised import SelfTrainingClassifier
import pandas as pd
import numpy as np
from sqlgenerator.helper_functions import build_and_run_base_query, get_predicate
import matplotlib.pyplot as plt

def predict_dtss(modelParams, sslThreshold):
    
    # get all the data for pseudo laeblling later
    data_all = pd.DataFrame(Adult.objects.all().values().using("adult"))

    # read in data
    with open("predict/predict_scripts/qbe_input/qbe_input_df.pickle", "rb") as fp:
        data_train = pickle.load(fp)

    # store user requested columns to return only those later 
    user_requested_cols = data_train.columns.values
    user_requested_cols = np.setdiff1d(user_requested_cols, "label")

    # use input extension mechanism (construct a base query to get examples from the DB that match input)
    base_query_output = build_and_run_base_query(data_train)    
    
    # get the training data from the user input as well as all data for pseudo labeling later
    x_train = base_query_output.loc[ : , base_query_output.columns != 'label']
    y_train = base_query_output.loc[ : , base_query_output.columns == 'label']

    # get all existing categorical fields and one hot encode them
    field_names = [f.name for f in Adult._meta.get_fields()]
    field_names.remove("id")
    char_fields = []
    field_types = [Adult._meta.get_field(f).get_internal_type() for f in field_names]
    for index, entry in enumerate(field_types):        
        if entry == "CharField":
            char_fields.append(field_names[index])
    
    # one hot encode training set
    x_train = pd.get_dummies(x_train, columns=char_fields)
    x_train = x_train.apply(pd.to_numeric)
    
    # one hot encode all data points from db and apply same col order
    data_all  = data_all.set_index('id')
    data_all = pd.get_dummies(data_all, columns=char_fields)
    data_all = data_all[x_train.columns]

    # add potentially missing cols after one hot encoding to smaller set (due to values not occurring)
    missing_cols = set(data_all.columns) - set(x_train.columns)
    for column in missing_cols:
        x_train[column] = 0

    # append all data to user input data, label unlabeled data with "-1" for SSL
    x_train_ssl = x_train.append(data_all)
    y_train_ssl = np.concatenate([y_train["label"], np.repeat(-1,(len(x_train_ssl)-len(x_train)))])
    # ----------------------------------------------------------------------------------------------------
    # PREPROCESSING DONE. Put aglorithm specific code here.
    # ----------------------------------------------------------------------------------------------------

    # ensure integers for some of the params
    modelParams['max_depth'] = int(modelParams['max_depth'])
    modelParams['min_samples_leaf'] = int(modelParams['min_samples_leaf'])
    modelParams['min_samples_split'] = int(modelParams['min_samples_split'])

    # create model instance
    base_model = tree.DecisionTreeClassifier(**modelParams)
    model = SelfTrainingClassifier(base_model, threshold=sslThreshold)
    # train the model on initial QBE input example
    model = model.fit(X=x_train_ssl, y=y_train_ssl)
    # get the predictions for all the data from the newly trained model
    preds = np.array(model.predict(data_all))

    # ----------------------------------------------------------------------------------------------------
    # Processing result
    # ----------------------------------------------------------------------------------------------------

    # check if predicted labels contain both labels or just the same label (More training data needed?)
    results_were_filtered = "False"
    if(1 in preds and 0 in preds):
        results_were_filtered = "True"

    # get only entries with label 1 (positive predictions)
    result = data_all.iloc[(preds > 0).tolist(),:]

    #store resutls as pickle file to load them later (but only user requested cols)
    result_cols = [col for col in result.columns if col.startswith(tuple(user_requested_cols))]
    result = result[result_cols]
    result.to_pickle("predict/predict_scripts/qbe_output/qbe_output_df.pickle")

    # save image of the tree
    tree.plot_tree(model.base_estimator_, feature_names=x_train.columns, impurity=False)
    plt.savefig("predict/predict_scripts/qbe_output/qbe_gdb_tree_image.png")
    plt.savefig("global/static/qbe_gdb_tree_image.png")


    # construct the query modeled by the tree
    where_conditions = get_predicate(model.base_estimator_, x_train.columns)
    query = "SELECT " + ", ".join(user_requested_cols) + "\n FROM adult WHERE \n" + str(where_conditions) + ";"
    with open("predict/predict_scripts/qbe_output/qbe_output_query.pickle", "wb+") as fp:
        pickle.dump(query, fp)

    return results_were_filtered