from algorithms.squid import prepare_squid
from experiments.experiment1 import run_experiment_1
from experiments.experiment2 import run_experiment_2
from experiments.experiment3 import run_experiment_3
from experiments.experiment4 import run_experiment_4
from experiments.experiment5 import run_experiment_5
import os, pickle, re
import pandas as pd
import numpy as np
from pathlib import Path

print("----------------------------------------")
print("Starting benchmark")
print("----------------------------------------")

# import the query set
query_set = pd.read_csv(
    Path.joinpath(Path(Path.cwd()).resolve().parents[0], str(os.getcwd()), 'data', 'query_set.csv'), 
    na_values="?").reset_index(drop=True)

# import the datasets
adult_dataset = pd.read_csv(
    Path.joinpath(Path(Path.cwd()).resolve().parents[0], str(os.getcwd()), 'data', 'raw', 'adult.csv')).reset_index(drop=True)
movie_dataset = pd.read_csv(
    Path.joinpath(Path(Path.cwd()).resolve().parents[0], str(os.getcwd()), 'data', 'raw', 'movies.csv'), sep="\t").reset_index(drop=True)
star_dataset = pd.read_csv(
    Path.joinpath(Path(Path.cwd()).resolve().parents[0], str(os.getcwd()), 'data', 'raw', 'stars.csv')).reset_index(drop=True)
star_dataset = star_dataset.drop(columns=["objID", "specobjID"])
movie_dataset = movie_dataset.drop(columns=["lead_gender"])
# replace inch and 
movie_dataset = movie_dataset.sample(frac=0.1, random_state=3549)
movie_dataset['lead_height'] = movie_dataset['lead_height'].str.replace("'",'ft')
movie_dataset['lead_height'] = movie_dataset['lead_height'].str.replace('"','in')
movie_dataset['lead_age'] = movie_dataset['lead_age'].str.replace('80+','gt80')
movie_dataset['lead_age'] = movie_dataset['lead_age'].str.replace('+','')
movie_dataset['title'] = movie_dataset['title'].str.replace('Ä','A')
movie_dataset['title'] = movie_dataset['title'].str.replace(pat=r'[^A-Za-z0-9\>\<\=\-\?\s]',repl='', regex=True)
movie_dataset['lead_actor'] = movie_dataset['lead_actor'].str.replace(pat=r'[^A-Za-z0-9\>\<\=\-\?\s]',repl='', regex=True)
movie_dataset['language'] = movie_dataset['language'].str.replace(pat=r'[^\w0-9\>\<\=\-\?\s]',repl='', regex=True)
movie_dataset['lead_country'] = movie_dataset['lead_country'].str.replace(pat=r'[^\w0-9\>\<\=\-\?\s]',repl='', regex=True)
movie_dataset.to_csv("data/raw/movies_small.csv", index=False)
stop
datasets = [adult_dataset, movie_dataset, star_dataset]

# preprocess and col names
for dataset in datasets:
    dataset.columns = [c.replace('.', '') for c in dataset.columns]
    na_fill = {c:dataset[c].mode()[0] for c in dataset.select_dtypes('O')}
    dataset = dataset.fillna(value= na_fill)

adult_dataset = datasets[0].rename(columns={"educational-num": "educationnum", "capital-gain": "capitalgain", "capital-loss": "capitalloss", "marital-status": "maritalstatus","hours-per-week": "hoursperweek","native-country": "nativecountry"})
movie_dataset = datasets[1].rename(columns={"production_year": "productionyear", "lead_actor": "leadactor", "lead_country": "leadcountry", "lead_birthyear" : "leadbirthyear", "lead_age": "leadage", "lead_height": "leadheight"})
star_dataset = datasets[2].rename(columns={"run_ID": "runID", "rerun_ID": "rerunID", "cam_col": "camcol", "field_ID": "fieldID", "spec_obj_ID": "specobjID", "fiber_ID": "fiberID",})

# precompute the alpha db's for SQuID
print("Starting SQuID pre-processing...")
prepare_squid(adult_dataset, "adult")
prepare_squid(movie_dataset, "movies")
prepare_squid(star_dataset, "stars")

# precompute one hot encodings for categorical columns
print("Starting one-hot encoding datasets...")
adult_dataset_onehot = pd.get_dummies(adult_dataset, columns=["workclass","education","maritalstatus","occupation","relationship","race","gender","nativecountry","income"])
movie_dataset_onehot = pd.get_dummies(movie_dataset, columns=["title","genre","language","leadactor","leadcountry","leadage","leadheight"])
star_dataset_onehot = pd.get_dummies(star_dataset, columns=["objclass"])

# remove forbidden characters in column names generated from one hot encoding
movie_dataset_onehot.columns = movie_dataset_onehot.columns.str.replace('< ', 'lt')
movie_dataset_onehot.columns = movie_dataset_onehot.columns.str.replace('<', '')
movie_dataset_onehot.columns = movie_dataset_onehot.columns.str.replace('> ', 'gt')
movie_dataset_onehot.columns = movie_dataset_onehot.columns.str.replace(' ', '')
movie_dataset_onehot.columns = [re.sub("[^a-zA-Z0-9\-\>\<\=\-\?\_]","",entry) for entry in movie_dataset_onehot.columns]
duplicate_columns_index = movie_dataset_onehot.columns.duplicated()
movie_dataset_onehot = movie_dataset_onehot.drop(movie_dataset_onehot.columns[duplicate_columns_index], axis=1)

# store precomputed one-hot-encoded dataframes if needed
# adult_dataset_onehot.to_csv("data/one_hot_encoded/onehot_adult.csv", index=False)
# movie_dataset_onehot.to_csv("data/one_hot_encoded/onehot_movies.csv", index=False)
# star_dataset_onehot.to_csv("data/one_hot_encoded/onehot_stars.csv", index=False)

# First experiment: Baseline. Supervised Learning, Balanced Dataset, No Base Query extension for DT, RF, GB, SQuID
experiment_1 = run_experiment_1(query_set, [adult_dataset, movie_dataset, star_dataset], [adult_dataset_onehot, movie_dataset_onehot, star_dataset_onehot])
experiment_1.to_csv('data/results/experiment_1.csv', index=False)
print(experiment_1)
# Second experiment: Supervised Learning, Balanced Dataset, with Base Query extension for DT, RF, GB, SQuID
experiment_2 = run_experiment_2(query_set, [adult_dataset, movie_dataset, star_dataset], [adult_dataset_onehot, movie_dataset_onehot, star_dataset_onehot])
experiment_2.to_csv('data/results/experiment_2.csv', index=False)
print(experiment_2)
# Third experiment: Supervised Learning, unbalanced Dataset, with Base Query extension for DT, RF, GB
experiment_3 = run_experiment_3(query_set, [adult_dataset, movie_dataset, star_dataset], [adult_dataset_onehot, movie_dataset_onehot, star_dataset_onehot])
experiment_3.to_csv('data/results/experiment_3.csv', index=False)
print(experiment_3)
# Fourth experiment: Semi-Supervised Learning, Balanced Dataset, with Base Query extension for DT, RF, GB
experiment_4 = run_experiment_4(query_set, [adult_dataset, movie_dataset, star_dataset], [adult_dataset_onehot, movie_dataset_onehot, star_dataset_onehot])
experiment_4.to_csv('data/results/experiment_4.csv', index=False)
print(experiment_4)
# Fifth experiment: Semi-Supervised Learning, unbalanced Dataset, with Base Query extension for DT, RF, GB
experiment_5 = run_experiment_5(query_set, [adult_dataset, movie_dataset, star_dataset], [adult_dataset_onehot, movie_dataset_onehot, star_dataset_onehot])
experiment_5.to_csv('data/results/experiment_5.csv', index=False)
print(experiment_5)
