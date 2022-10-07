import pandas as pd
import mysql.connector
import pickle
import numpy as np
from scipy.spatial import ConvexHull
from regression_helper_functions import get_min_max_sample, vectorize_list_items, pad_missing_cols, clean_col_names
from gensim.models import Word2Vec

# params
db_name = "adult"
table_name = "adult"
file_output_name = "talos"
categorical_cols = ["workclass","education","maritalstatus","occupation","relationship","race","gender","nativecountry","income"]
dataset_path = "data/ds-generator/adult/talos"
w2v_size = len(categorical_cols)  #size of the word vectors generated later by word2vec
conv_hull_precision = 1 # fraction of points used to find conv. hull. 0 to 1. Smaller values are faster, larger more precise.

# load SQL queries to be used
sql_list = [
            "SELECT age, hoursperweek FROM adult WHERE age BETWEEN 10 AND 20 AND hoursperweek < 30;",
            "SELECT age, hoursperweek FROM adult WHERE age BETWEEN 20 AND 30 AND hoursperweek < 30;",
            "SELECT age, hoursperweek FROM adult WHERE age BETWEEN 30 AND 40 AND hoursperweek < 30;",
            "SELECT age, hoursperweek FROM adult WHERE age BETWEEN 40 AND 50 AND hoursperweek < 30;",
            "SELECT age, hoursperweek FROM adult WHERE age BETWEEN 50 AND 60 AND hoursperweek < 30;",
            "SELECT age, hoursperweek FROM adult WHERE age BETWEEN 60 AND 70 AND hoursperweek < 30;",
            "SELECT age, hoursperweek FROM adult WHERE age BETWEEN 70 AND 80 AND hoursperweek < 30;",
            "SELECT age, hoursperweek FROM adult WHERE age BETWEEN 80 AND 100 AND hoursperweek < 30;",
            "SELECT age, hoursperweek FROM adult WHERE age BETWEEN 10 AND 20 AND hoursperweek > 30;",
            "SELECT age, hoursperweek FROM adult WHERE age BETWEEN 20 AND 30 AND hoursperweek > 30;",
            "SELECT age, hoursperweek FROM adult WHERE age BETWEEN 30 AND 40 AND hoursperweek > 30;",
            "SELECT age, hoursperweek FROM adult WHERE age BETWEEN 40 AND 50 AND hoursperweek > 30;",
            "SELECT age, hoursperweek FROM adult WHERE age BETWEEN 50 AND 60 AND hoursperweek > 30;",
            "SELECT age, hoursperweek FROM adult WHERE age BETWEEN 60 AND 70 AND hoursperweek > 30;",
            "SELECT age, hoursperweek FROM adult WHERE age BETWEEN 70 AND 80 AND hoursperweek > 30;",
            "SELECT age, hoursperweek FROM adult WHERE age BETWEEN 80 AND 100 AND hoursperweek > 30;",            
            ]

# connect to MySQL DB
# Might need to run "sudo service mysql start" beforehand in WSL
cnx = mysql.connector.connect(user='python_user', password='pthon_user_password',
                              host='localhost',
                              database='adult')
cursor = cnx.cursor()
full_table = pd.read_sql("SELECT * FROM " + table_name +";", cnx)

# empty list of all results
dataset = []

# Iterate over list of SQL queries
for index, query in enumerate(sql_list):
    # Run Query
    result_fullset = pd.read_sql(query, cnx)  
    # column names that the user wanted
    result_cols = result_fullset.columns.values
    # try to find extreme points of the result set the user wanted (convex hull)
    unique_rows = result_fullset.drop_duplicates()
    # unique_rows_sample = unique_rows.sample(frac = conv_hull_precision, random_state=5258)  
    unique_rows_sample = get_min_max_sample(unique_rows, frac = conv_hull_precision, random_state=5258)
    hull_obj = ConvexHull(unique_rows_sample, qhull_options="Qt")
    result_fullset_hull = pd.concat([unique_rows_sample.iloc[[i]] for i in hull_obj.vertices])
    # Get random subset of results to use as QBE input (size of the numbers of vertices in conv. hull)
    result_subset = get_min_max_sample(unique_rows, n=len(result_fullset_hull.index), random_state=5258) 
    # bring training data and labels in same order 
    result_subset.sort_index(inplace=True)
    result_fullset_hull.sort_index(inplace=True)
    # and append results to list of all results
    dataset.append([query, result_subset, result_fullset, result_cols, result_fullset_hull])

#close DB connection
cnx.close()

# vectorize text data in dataset
text_cols = [list(row) for row in full_table[categorical_cols].to_numpy()]
model = Word2Vec(sentences=text_cols, vector_size=w2v_size, min_count=1, workers=4)
dataset = [vectorize_list_items(entry, categorical_cols, model, w2v_size) for entry in dataset]

# pad the qbe example with the columns missing compared to the full table. Training sets all need to have the same number of cols
dataset = [pad_missing_cols(entry, full_table.columns.values, categorical_cols, w2v_size) for entry in dataset]

# clean col names from special characters created by one hot encoding
dataset = [clean_col_names(entry) for entry in dataset]

print("################")
print(dataset)

# Store results to file system as a pickled list of lists each containing:
# SQL query, QBE subset, full result set, columns requested, row IDs of results
with open("data/ds-generator/"+ db_name + "/" + file_output_name +"_preprocessed", "wb") as fp:
    pickle.dump(dataset, fp)

