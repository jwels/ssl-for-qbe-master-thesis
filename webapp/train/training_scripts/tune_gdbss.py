import pickle
from sqlgenerator.models import TrainingData
# from train.models import TrainedModel
import xgboost as xgb
from sklearn.semi_supervised import SelfTrainingClassifier
import pandas as pd
import numpy as np
import statistics
from sklearn.metrics import f1_score
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from functools import partial

def hyperopt_gdbss(model):

    space = {
        'n_estimators': hp.uniformint('n_estimators', 3, 300),
        'eta': hp.quniform('eta', 0.025, 1.5, 0.025),
        'max_depth': hp.uniformint('max_depth', 1, 14),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
        'objective': 'binary:logistic'
    }

    # Use the fmin function from Hyperopt to find the best hyperparameters
    trials = Trials()
    best = fmin(partial(tune, modelOptions=model), space, trials=trials, algo=tpe.suggest, max_evals=model.maxEvals)
    max_f1_score_trial = {'result':{'total_f1_score_val': -1}}
    for entry in trials:
        if(entry.get('result', {}).get('total_f1_score_val')>max_f1_score_trial.get('result', {}).get('total_f1_score_val')):
            max_f1_score_trial = entry
    return [max_f1_score_trial.get('result', {}).get('total_f1_score_test'), max_f1_score_trial.get('result', {}).get('total_f1_score_val'), best]

def tune(params, modelOptions):


    # get Training Data and split into train and test set
    data_all = TrainingData.objects.all()

    # create model instance
    base_model = xgb.XGBClassifier(**params)
    ssl_base_model = xgb.XGBClassifier(**params)
    model = SelfTrainingClassifier(ssl_base_model, threshold=modelOptions.removeLowestCertaintyPercentage)

    # instantiate variables
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
        x_test = x_test[col_order]
        x_val = x_val[col_order]
        x_train = x_train.drop(columns=["id"])
        x_test = x_test.drop(columns=["id"])
        x_val = x_val.drop(columns=["id"])

        # skip when y_val too small or y_test has only unique labels
        if (len(y_val)<5 or len(set(y_test))==1):
            continue

        # train the model on initial QBE input example to check F1 score without SSL
        base_model = base_model.fit(X=x_train, y=y_train)
        # get predictions
        preds_base_model = base_model.predict(x_test)
        # F1-Score on test set before semi-supervised learning
        f1_scores_test.append(f1_score(y_test, preds_base_model, zero_division=0))

        # train again with SSL this time, unlabeled data gets label -1
        x_train_ssl = x_train.append(x_test)
        y_train_ssl = np.concatenate([y_train, np.repeat(-1,len(x_test))])
        model = model.fit(X=x_train_ssl, y=y_train_ssl)

        # F1-Score on validation set after semi-supervised learning step
        preds = model.predict(x_val)
        f1_scores_val.append(f1_score(y_val, preds, zero_division=0))

    if(len(f1_scores_val)==0 or len(f1_scores_test)==0):
        total_f1_score_test = 0
        total_f1_score_val = 0
    else:
        total_f1_score_test = statistics.median(f1_scores_test)#sum(f1_scores_test)/len(f1_scores_test)
        total_f1_score_val = statistics.median(f1_scores_val)#sum(f1_scores_val)/len(f1_scores_val)    
   
    return {'loss': 1-total_f1_score_val, 'status': STATUS_OK, 'total_f1_score_test': total_f1_score_test, 'total_f1_score_val': total_f1_score_val}