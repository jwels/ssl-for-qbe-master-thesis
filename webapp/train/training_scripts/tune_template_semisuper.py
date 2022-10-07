import pickle
from sqlgenerator.models import TrainingData
import pandas as pd
import numpy as np
import statistics
from sklearn.metrics import f1_score
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from functools import partial

def hyperopt_templatess(model):

    # -------------------------------------------------------
    # TODO: 
    # 1) add your hyperparameters here
    # 2) add your model code below in "tune" function
    # -------------------------------------------------------

    space = {
        # -------------------------------------------------------
        # Put your Hyperparameters and ranges here using hyperopt syntax
        # Example:
        # 'n_estimators': hp.uniformint('n_estimators', 3, 300),
        # 'eta': hp.quniform('eta', 0.025, 1.5, 0.025),
        # -------------------------------------------------------
    }

    # Use the fmin function from Hyperopt to find the best hyperparameters
    trials = Trials()
    best = fmin(partial(tune, model=model), space, trials=trials, algo=tpe.suggest, max_evals=model.maxEvals)
    max_f1_score_trial = {'result':{'total_f1_score_val': -1}}
    for entry in trials:
        if(entry.get('result', {}).get('total_f1_score_val')>max_f1_score_trial.get('result', {}).get('total_f1_score_val')):
            max_f1_score_trial = entry
    return [max_f1_score_trial.get('result', {}).get('total_f1_score_test'), max_f1_score_trial.get('result', {}).get('total_f1_score_val'), best]

def tune(params, model):


    # get Training Data and split into train and test set
    data_all = TrainingData.objects.all()

    # -------------------------------------------------------
    # create your model instance before the loop
    # your_model = someClassifier(**params)
    # -------------------------------------------------------

    # instantiate variables
    counter = 0
    f1_scores_test = []
    f1_scores_val = []

    # import the data for current batch
    for entry in data_all:

        # read in data
        with open(entry.data_pickle_path, "rb") as fp:
            data_train = pickle.load(fp)

        # get the training, test and validation set and their labels
        x_train = data_train[0][0].loc[ : , data_train[0][0].columns != 'label']
        y_train = np.ravel(data_train[0][0].loc[ : , data_train[0][0].columns == 'label'])
        x_train = x_train.apply(pd.to_numeric)
        data_train[0][1] = data_train[0][1].reset_index(drop=True)
        data_train[0][1] = data_train[0][1].sample(frac=1)
        x_test = data_train[0][1].loc[ : , data_train[0][1].columns != 'label']
        y_test = data_train[0][1].loc[ : , data_train[0][1].columns == 'label']
        x_test = x_test.apply(pd.to_numeric)
        x_val = x_test.tail(round(len(x_test)*0.3))
        x_test = x_test.head(round(len(x_test)*0.7))
        y_val = np.ravel(y_test.tail(round(len(y_test)*0.3)))
        y_test = np.ravel(y_test.head(round(len(y_test)*0.7)))
        col_order = x_train.columns

        # skip when y_val too small or y_test has only unique labels
        if (len(y_val)<5 or len(set(y_test))==1):
            counter = counter + 1
            continue

        # -------------------------------------------------------
        # Put your first model fit (before SSL) here:
        pseudo_labels = "First Predictions before SSL"
        # Optionally: use model.removeLowestCertaintyPercentage to get user specified percentage to be removed
        # Then: remove the most uncertain percentage of predictions before next step
        # -------------------------------------------------------

        # F1-Score on test set before semi-supervised learning step
        f1_scores_test.append(f1_score(y_test, pseudo_labels, zero_division=0))

        # -------------------------------------------------------
        # Put your new model fit (SSL step) here:
        preds = "Your Model Predictions"
        # -------------------------------------------------------

        # append F1-Scores after SSL
        f1_scores_val.append(f1_score(y_val, preds, zero_division=0))

        counter = counter + 1 

    if(len(f1_scores_val)==0 or len(f1_scores_test)==0):
        total_f1_score_test = 0
        total_f1_score_val = 0
    else:
        total_f1_score_test = statistics.median(f1_scores_test)#sum(f1_scores_test)/len(f1_scores_test)
        total_f1_score_val = statistics.median(f1_scores_val)#sum(f1_scores_val)/len(f1_scores_val)    
   
    return {'loss': 1-total_f1_score_val, 'status': STATUS_OK, 'total_f1_score_test': total_f1_score_test, 'total_f1_score_val': total_f1_score_val}