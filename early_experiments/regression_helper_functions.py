import pandas as pd
import numpy as np
from gensim.models import Word2Vec

# Gets a list of [SQL, QBE Set, Full Result Set, label col names, label row IDs] and a list of cols. Returns the same list but one hot encoded all columns in cols.
def one_hot_encode_list_items(x, cols):
   subset_column_headers = list(x[1].columns.values)
   existing_categorical_cols = set(subset_column_headers).intersection(cols)
   return [x[0],pd.get_dummies(x[1], columns = existing_categorical_cols),pd.get_dummies(x[2], columns = existing_categorical_cols)]

# turns specified cols in the dataset (passed as x) into vector representations using a word2vec model
def vectorize_list_items(x, cols, model, w2v_size):
   subset_column_headers = list(x[1].columns.values)
   existing_text_cols = set(subset_column_headers).intersection(cols)
   # iterate over every text column that should be vectorized
   for col in existing_text_cols:       
      new_cols = []
      # generate new column names
      for index in range(0,w2v_size):
         new_col = col +"_wv" + str(index)
         new_cols.append(new_col)
      # fill new columns with the data from word2vec
      new_cols_data_1 = pd.DataFrame(model.wv[x[1][col]], columns=new_cols)
      new_cols_data_2 = pd.DataFrame(model.wv[x[2][col]], columns=new_cols)

      # copy index from the input to enable joining (inserting the new cols)
      new_cols_data_1 = new_cols_data_1.set_index(x[1].index)
      x[1] =  x[1].join(new_cols_data_1)
      x[2] =  x[2].join(new_cols_data_2)
   
      # drop columns that have now been replaced by word2vec data
      x[1] = x[1].drop(col, 1)
      x[2] = x[2].drop(col, 1)
      # sort columns
      x[1] = x[1].reindex(sorted(x[1].columns), axis=1)
      x[2] = x[2].reindex(sorted(x[2].columns), axis=1)
  
   return [x[0], x[1], x[2]]

# pad dataset with empty columns to make it the same size for every training iteration. Also sort columns to receive them always in the same order.
def pad_missing_cols(x, all_cols, vectorized_cols, w2v_size):
   
   # iterate over every text column that have been vectorized
   vec_cols = []
   for col in vectorized_cols: 
      # generate new column names
      for index in range(0,w2v_size):
         vec_cols.append(col +"_wv" + str(index))

   existing_cols = list(x[1].columns.values)
   missing_cols = list(set(all_cols).union(set(vec_cols)).difference(existing_cols))
   
   for col in missing_cols:
      x[1][col] = 0
      x[2][col] = 0

   # drop columns that have now been replaced by word2vec data
   x[1] = x[1].drop(vectorized_cols, 1)
   # sort columns
   x[1] = x[1].reindex(sorted(x[1].columns), axis=1)
   x[2] = x[2].reindex(sorted(x[2].columns), axis=1)

   return x


# pad dataset with empty columns to make it the same size for every training iteration. Also sort columns to receive them always in the same order.
def old_pad_missing_cols(x, all_cols):
   existing_cols = list(x[1].columns.values)
   missing_cols = list(set(all_cols).difference(existing_cols))
   
   for col in missing_cols:
      x[1][col] = pd.Series()
      x[2][col] = pd.Series()

   # sort columns
   x[1] = x[1].reindex(sorted(x[1].columns), axis=1)
   x[2] = x[2].reindex(sorted(x[2].columns), axis=1)

   return x

# clean col names from special characters created by one hot encoding
def clean_col_names(x):
   x[1].columns = x[1].columns.str.replace('<=', 'under')
   x[2].columns = x[2].columns.str.replace('<=', 'under')
   x[1].columns = x[1].columns.str.replace('>', 'over')
   x[2].columns = x[2].columns.str.replace('>', 'over')
   return x

# function to get subset of a table containing the most extreme points of each dimension
def get_min_max_sample(x, n = None, frac = None, random_state=5258):
   result_set = pd.DataFrame()
   for column in x.columns.values:
      x = x.sort_values(by=[column], ascending=True)
      result_set = result_set.append(x.head(1))
      result_set = result_set.append(x.tail(1))
   if(frac!=None):
      result_set = result_set.append(x.sample(frac = frac, random_state = random_state))
   if(n!=None):
      result_set = result_set.append(x.sample(n = n-len(result_set.index), random_state = random_state))
   return result_set