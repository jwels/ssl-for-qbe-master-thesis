import pickle
import statistics
from sqlgenerator.models import TrainingData
# from train.models import TrainedModel
from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

def hyperopt_dt(model):

    space = {
        'criterion': 'gini',
        'splitter': 'best',
        'max_depth': hp.uniformint('max_depth', 1, 14),
        'min_samples_split': hp.uniformint('min_samples_split', 2, 14),
        'min_samples_leaf': hp.uniformint('min_samples_leaf', 1, 14)
    }

    # Use the fmin function from Hyperopt to find the suitable hyperparameters
    trials = Trials()
    best = fmin(tune, space, trials=trials, algo=tpe.suggest, max_evals=model.maxEvals)
    max_f1_score_trial = {'result':{'total_f1_score_test': -1}}
    for entry in trials:
        if(entry.get('result', {}).get('total_f1_score_test')>max_f1_score_trial.get('result', {}).get('total_f1_score_test')):
            max_f1_score_trial = entry
    return [max_f1_score_trial.get('result', {}).get('total_f1_score_test'), max_f1_score_trial.get('result', {}).get('total_f1_score_val'), best]

def tune(params):
    

    # get Training Data and split into train and test set
    data_all = TrainingData.objects.all()

    # create model instance
    model = tree.DecisionTreeClassifier(**params)

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
        # le = LabelEncoder()
        # le.fit_transform(data_train[0][1].loc[ : , data_train[0][1].columns == 'label'])
        # le.transform(data_train[0][0].loc[ : , data_train[0][0].columns == 'label'])
        x_train = data_train[0][0].loc[ : , data_train[0][0].columns != 'label']
        y_train = np.ravel(data_train[0][0].loc[ : , data_train[0][0].columns == 'label'])
        x_train = x_train.apply(pd.to_numeric)
        x_test = data_train[0][1].loc[ : , data_train[0][1].columns != 'label']
        y_test = data_train[0][1].loc[ : , data_train[0][1].columns == 'label']
        x_test = x_test.apply(pd.to_numeric)
        col_order = x_train.columns
        x_test = x_test[col_order]
        x_train = x_train.drop(columns=["id"])
        x_test = x_test.drop(columns=["id"])


        # print("First training, Iteration: " + str(counter))
        # x_train = x_train[col_order]

        # train the model on initial QBE input example
        model = model.fit(X=x_train, y=y_train)
        # F1-Score on test set before semi-supervised learning step
        pseudo_labels = model.predict(x_test)
        f1_scores_test.append(f1_score(y_test, pseudo_labels, zero_division=0))

        counter = counter + 1

    if(len(f1_scores_test)==0):
        total_f1_score_test = 0
        total_f1_score_val = None
    else:
        total_f1_score_test = statistics.median(f1_scores_test)#sum(f1_scores_test)/len(f1_scores_test)
        total_f1_score_val = None
   
    # [sum(f1_scores_test)/len(f1_scores_test),sum(f1_scores_val)/len(f1_scores_val), json.dumps(params)]
    return {'loss': 1-total_f1_score_test, 'status': STATUS_OK, 'total_f1_score_test': total_f1_score_test, 'total_f1_score_val': total_f1_score_val}