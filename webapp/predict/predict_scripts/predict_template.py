import pickle
from sqlgenerator.models import Adult
import xgboost as xgb
import pandas as pd
import numpy as np
from sqlgenerator.helper_functions import build_and_run_base_query

def predict_template(modelParams):
    
    # get all the data for pseudo laeblling later
    data_all = pd.DataFrame(Adult.objects.all().values().using("adult"))

    # read in data
    with open("predict/predict_scripts/qbe_input/qbe_input_df.pickle", "rb") as fp:
        data_train = pickle.load(fp)

    # store user requested columns to return only those later 
    user_requested_cols = data_train.columns
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

    # ----------------------------------------------------------------------------------------------------
    # PREPROCESSING DONE. Put aglorithm specific code here.
    # ----------------------------------------------------------------------------------------------------

    # put algorithm code here
    # use x_train and y_train for fitting the model
    # store predicitions as dataframe "preds"
    preds = pd.DataFrame()

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

    return results_were_filtered