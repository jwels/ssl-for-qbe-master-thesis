import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import f1_score, recall_score, precision_score
import re


# calculate f1 score depending on output format: squid output or labels (scikit-learn)
def calculate_f1_score(predictions, ground_truth, all, squid = False):
    
    # get 0 or 1 labels for all entries, depending on if they are in the result or not
    if(squid):
        y_predictions_list = [1 if entry in predictions.index else 0 for entry in all.index]
        y_predictions = pd.Series(y_predictions_list, index=all.index)
    else:
        y_predictions = predictions

    # same for the ground truth
    y_truth_list = [1 if entry in ground_truth.index else 0 for entry in all.index]
    y_truth = pd.Series(y_truth_list, index=all.index)

    f1 = f1_score(y_true=y_truth, y_pred=y_predictions)
    recall = recall_score(y_true=y_truth, y_pred=y_predictions)
    precision = precision_score(y_true=y_truth, y_pred=y_predictions)

    return [f1, recall, precision]


# take a query string and transform the categorical columns to query condtions on one-hot-encoded data
def translate_query_for_onehot_table(query):
    pattern = "(workclass|education|maritalstatus|occupation|relationship|race|gender|nativecountry|income|title|genre|language|leadactor|leadgenre|leadcountry|leadbirthyear|leadage|leadheight|objclass)\s?(==|!=)\s?\"([\w0-9\>\<\=\-\?\s]+)\""
    # if categoircal column name contained
    while(not re.search(pattern, query) is None):        
        # get capture groups using above regex pattern
        regex_results = re.search(pattern, query)
        if(regex_results.group(2)=="=="):
            new_query = query[:regex_results.span()[0]] + '`' + regex_results.group(1) + '_' + regex_results.group(3).replace(" ", "") + '` == 1' + query[regex_results.span()[1]:]
        else:
            new_query = query[:regex_results.span()[0]] + '`' + regex_results.group(1) + '_' + regex_results.group(3).replace(" ", "") + '` == 0' + query[regex_results.span()[1]:]
        query = new_query

    return str(query)


# make SQL susbtring from df column, used in build_and_run_base_query
def column_to_sql_string(df, col, condition, single=False):
   # check if editing a single column or not
   if(single): conjunction = ""
   else: conjunction = " and "
   # numerical column
   if(pd.to_numeric(df[col], errors='coerce').notnull().all()):    
      col_max = df[col].max()
      col_min = df[col].min()
      if(col_min != col_max):

         condition = condition + conjunction + \
               str(col) + " >= " + str(col_min) + " and " \
               + str(col) + " <= " + str(col_max)
      else:
         condition = condition + conjunction + \
               str(col) + " == " + str(col_min)
   # categorical column
   else:
      unique_values = df[col].unique()
      categorical_condition = conjunction + '(' + \
         str(col) + ' == "' + str(unique_values[0]) + '"'
      for value in unique_values[1:]:
         categorical_condition = categorical_condition + \
               ' or ' + str(col) + ' == "' + str(value) + '"'
      condition = condition + categorical_condition + ")"
   return condition

# get QBE input example, construct base queries from that. Return extended example.
def build_base_query(example):
    # build query conditions for positive query
    index = 0
    pos_condition = ""
    # Case 1: only one column:
    if(len(example.columns) == 1):
      pos_condition = column_to_sql_string(example, example.columns.values[0], "", single=True)
    # Case 2: more than one column
    else:
      # get number of unique column values
      positives_unique = example.nunique()
      # check if a numerical column has been selected for grouping (positive set)
      pos_contains_categoric_col = False
      pos_last_option_found = False
      for col in example.columns:
         # check if there is atleast one categoric column
         if(not pd.to_numeric(example[col], errors='coerce').notnull().all()):
            pos_contains_categoric_col = True
            # if categorical col has only one unique entry, store as last grouping option as more than 1 is preferable
            if(positives_unique[col] == 1):
                pos_last_option_found = True
                pos_last_group_option = col
                positives_unique = positives_unique.drop(col)
      # get index of column with least unique values if not only one exists
      if(len(positives_unique)>0):
         pos_least_unique_col = positives_unique.idxmin()
      else:
         pos_least_unique_col = pos_last_group_option
      # select new column as group by target until non-numeric column was selected (positive set)
      while(pd.to_numeric(example[pos_least_unique_col], errors='coerce').notnull().all() and pos_contains_categoric_col):
         positives_unique = positives_unique.drop(pos_least_unique_col)
         # if no alternative columns are left, take last categorical grouping option from above
         if(len(positives_unique)==0 and pos_last_option_found):
            pos_least_unique_col = pos_last_group_option
            break         
         pos_least_unique_col = positives_unique.idxmin()

      # group df by "least unique" or most uniform column
      pos_grouped_by = example.groupby(pos_least_unique_col)      

      # loop over postive set's columns
      for key, item in pos_grouped_by:
         # get the dataframe from the pandas grouped-by object
         pos_df = pos_grouped_by.get_group(key)
         # list of cols in DF other than the one grouped by
         pos_cols = np.setdiff1d(pos_df.columns.values, pos_least_unique_col)
         # add the column used for grouping to the condition, with/without quotation marks depending on dtype
         if(is_numeric_dtype(pos_df[pos_least_unique_col])):
            pos_condition = pos_condition + \
                     "(" + str(pos_least_unique_col) + ' == ' + \
                     str(pos_df[pos_least_unique_col].values[0]) + ''
         else:
            pos_condition = pos_condition + \
                     "(" + str(pos_least_unique_col) + ' == "' + \
                     str(pos_df[pos_least_unique_col].values[0]) + '"'
         for col in pos_cols:
               pos_condition = column_to_sql_string(pos_df, col, pos_condition)
         pos_condition = pos_condition + ")"
            # close this condition for least_unique_col with OR
         if(index < (pos_grouped_by.ngroups - 1)):
               pos_condition = pos_condition + " or "
         index = index + 1

    return pos_condition

    # # Run both base queries
    # pos_query_result = all_data.query(pos_condition)    

    # # append, shuffle and set index
    # output = pos_query_result
    # output = output.dropna()
    # output = output.sample(frac=1).reset_index(drop=True)
    # # output = output.set_index('id')

    # # return both query results as one df with labels
    # return output