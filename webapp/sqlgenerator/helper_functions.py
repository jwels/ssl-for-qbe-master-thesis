from pdb import post_mortem
import pandas as pd
import numpy as np
from django.db import connections
import pickle
import re

# check an input table for range syntax [x,y] and replace those rows with two rows containing x and y respectively
def parse_syntax_from_input(data):
   # get list of cells containing the syntax for specyfing a range of numbers
   cell_list = list(data.applymap(lambda x: re.match("\[{1}\d*,\s*\d*\]{1}", str(x))).values.nonzero())

   # while there are cells to change
   while(len(cell_list[0])>0):
      # get the row to edit
      row = data.iloc[cell_list[0][0],]
      # read out the values in brackets [x,y]
      values = re.search('\[{1}(.*),\s*(.*)\]{1}', str(row.iloc[cell_list[1][0]]), re.IGNORECASE)
      value_min = values.group(1)
      value_max = values.group(2)
      # make two copies of the row and edit with read values
      row_min = row.copy()
      row_min.update(pd.Series([value_min], index=[data.columns[cell_list[1][0]]]))
      row_max = row.copy()
      row_max.update(pd.Series([value_max], index=[data.columns[cell_list[1][0]]]))
      # append to dataframe
      data = data.append(pd.DataFrame(row_min).transpose()).reset_index(drop=True)
      data = data.append(pd.DataFrame(row_max).transpose()).reset_index(drop=True)
      # drop original row
      data = data.drop(cell_list[0][0]).reset_index(drop=True)
      # refresh the list of cells containing syntax every iteration
      cell_list = list(data.applymap(lambda x: re.match("\[{1}\d*,\s*\d*\]{1}", str(x))).values.nonzero())

   
   # for debugging
   with open("predict/predict_scripts/qbe_output/base_query_syntax.pickle", "wb+") as fp:
      pickle.dump(data, fp)

   return data

# make SQL susbtring from df column, used in build_and_run_base_query
def column_to_sql_string(df, col, condition, single=False):
   # check if editing a single column or not
   if(single): conjunction = ""
   else: conjunction = " AND "
   # numerical column
   if(pd.to_numeric(df[col], errors='coerce').notnull().all()):
      col_max = df[col].max()
      col_min = df[col].min()
      if(col_min != col_max):

         condition = condition + conjunction + \
               str(col) + " BETWEEN " + \
               str(col_min) + " AND " + str(col_max)
      else:
         condition = condition + conjunction + \
               str(col) + " = " + str(col_min)
   # categorical column
   else:
      unique_values = df[col].unique()
      # if there are more than 1 unique value, wrap in parenthesis
      if(len(unique_values)>1):
        open_bracket = "("
        close_bracket = ")"
      else:
         open_bracket = ""
         close_bracket = ""
      categorical_condition = conjunction + open_bracket + str(
         col) + " = '" + unique_values[0] + "'"
      for value in unique_values[1:]:
         categorical_condition = categorical_condition + \
               " OR " + str(col) + " = '" + value + "'"
      condition = condition + categorical_condition + close_bracket
   if(single): condition = condition + ";"
   return condition

# get QBE input example, construct base queries from that. Return extended example.
def build_and_run_base_query(example):
    # build query conditions for positive query
    index = 0
    pos_condition = ""
    db_name = "adult"
    # split between pos. and neg. labels
    positives = example[example["label"] == 1]
    negatives = example[example["label"] == 0]
    positives = positives.drop(columns=["label"])
    negatives = negatives.drop(columns=["label"])
    # Filter for range syntax (eg [20,40]) and convert to additional examples
    positives = parse_syntax_from_input(positives)
    negatives = parse_syntax_from_input(negatives)
    with open("predict/predict_scripts/qbe_output/temp.pickle", "wb+") as fp:
        pickle.dump(positives, fp)
    # Case 1: only one column:
    if(len(positives.columns) == 1):
      pos_condition = column_to_sql_string(positives, positives.columns.values[0], "", single=True)
      neg_condition = column_to_sql_string(negatives, negatives.columns.values[0], "", single=True)
    # Case 2: more than one column
    else:
      # get number of unique column values
      positives_unique = positives.nunique()
      negatives_unique = negatives.nunique()
      # check if a numerical column has been selected for grouping (positive set)
      pos_contains_categoric_col = False
      pos_last_option_found = False
      for col in positives.columns:
         # check if there is atleast one categoric column
         if(not pd.to_numeric(positives[col], errors='coerce').notnull().all()):
            pos_contains_categoric_col = True
            # if categorical col has only one unique entry, store as last grouping option as more than 1 is preferable
            if(positives_unique[col] == 1):
                pos_last_option_found = True
                pos_last_group_option = col
                positives_unique = positives_unique.drop(col)
      # check if a numerical column has been selected for grouping (negative set)
      neg_contains_categoric_col = False
      neg_last_option_found = False
      for col in negatives.columns:
         # check if there is atleast one categoric column
         if(not pd.to_numeric(negatives[col], errors='coerce').notnull().all()):
            neg_contains_categoric_col = True
            # if categorical col has only one unique entry, store as last grouping option as more than 1 is preferable
            if(negatives_unique[col] == 1):
                neg_last_option_found = True
                neg_last_group_option = col
                negatives_unique = negatives_unique.drop(col)
      # get index of column with least unique values
      pos_least_unique_col = positives_unique.idxmin()
      neg_least_unique_col = negatives_unique.idxmin()
      # select new column as group by target until non-numeric column was selected (positive set)
      while(pd.to_numeric(positives[pos_least_unique_col], errors='coerce').notnull().all() and pos_contains_categoric_col):
         positives_unique = positives_unique.drop(pos_least_unique_col)
         # if no alternative columns are left, take last categorical grouping option from above
         if(len(positives_unique)==0 and pos_last_option_found):
            pos_least_unique_col = pos_last_group_option
            break         
         pos_least_unique_col = positives_unique.idxmin()
      # select new column as group by target until non-numeric column was selected (negative set)
      while(pd.to_numeric(negatives[neg_least_unique_col], errors='coerce').notnull().all() and neg_contains_categoric_col):
         negatives_unique = negatives_unique.drop(neg_least_unique_col)
         # if no alternative columns are left, take last categorical grouping option from above
         if(len(negatives_unique)==0 and neg_last_option_found):
            neg_least_unique_col = neg_last_group_option
            break  
         neg_least_unique_col = negatives_unique.idxmin()
      # group df by "least unique" or most uniform column
      pos_grouped_by = positives.groupby(pos_least_unique_col)
      neg_grouped_by = negatives.groupby(neg_least_unique_col)

      # loop over postive set's columns
      for key, item in pos_grouped_by:
         # get the dataframe from the pandas grouped-by object
         pos_df = pos_grouped_by.get_group(key)
         # list of cols in DF other than the one grouped by
         pos_cols = np.setdiff1d(pos_df.columns.values, pos_least_unique_col)
      
         pos_condition = pos_condition + \
                  "(" + str(pos_least_unique_col) + " = '" + \
                  str(pos_df[pos_least_unique_col].values[0]) + "'"
         for col in pos_cols:
               pos_condition = column_to_sql_string(pos_df, col, pos_condition)
         pos_condition = pos_condition + ")"
            # close this condition for least_unique_col with OR
         if(index < (pos_grouped_by.ngroups - 1)):
               pos_condition = pos_condition + " OR "
         else:
               pos_condition = pos_condition + ";"
         index = index + 1

      # build query conditions for negative query
      index = 0
      neg_condition = ""
      # loop over postive set's columns
      for key, item in neg_grouped_by:
         neg_df = neg_grouped_by.get_group(key)
         neg_cols = np.setdiff1d(neg_df.columns.values, neg_least_unique_col)      
         neg_condition = neg_condition + \
            "(" + str(neg_least_unique_col) + " = '" + \
            str(neg_df[neg_least_unique_col].values[0]) + "'"
         for col in neg_cols:
            neg_condition = column_to_sql_string(neg_df, col, neg_condition)
         neg_condition = neg_condition + ")"
         # close this condition for least_unique_col with OR
         if(index < (neg_grouped_by.ngroups - 1)):
               neg_condition = neg_condition + " OR "
         else:
               neg_condition = neg_condition + ";"
         index = index + 1

    # append base select to SQLs
    pos_sql = "SELECT * FROM " + db_name + " WHERE " + pos_condition
    neg_sql = "SELECT * FROM " + db_name + " WHERE " + neg_condition

    # Run both base queries
    pos_query_result = pd.read_sql(pos_sql, connections[db_name])    
    neg_query_result = pd.read_sql(neg_sql, connections[db_name])

    # format constructed base queries to look nice on UI
    pos_sql = pos_sql.replace("WHERE", "WHERE \n")
    pos_sql = pos_sql.replace("(", "( \n \t")    
    pos_sql = re.sub("(?<!BETWEEN\s\d\d\s)AND", "\n \t AND", pos_sql)
    # pos_sql = pos_sql.replace("AND", "\n \t AND")
    pos_sql = pos_sql.replace(")", "\n )")
    pos_sql = pos_sql.replace("OR", "\n OR")
    # store constructed base queries for UI
    with open("predict/predict_scripts/qbe_output/base_query_pos.pickle", "wb+") as fp:
        pickle.dump(pos_sql, fp)
    with open("predict/predict_scripts/qbe_output/base_query_neg.pickle", "wb+") as fp:
        pickle.dump(neg_sql, fp)



    # restore balance to the datasets: drop rows from neg. that are in pos. and balance sizes
    neg_query_result = neg_query_result[~(neg_query_result["id"].isin(list(pos_query_result["id"])))]
    if(len(pos_query_result) < len(neg_query_result)):
        pos_query_result = pos_query_result.sample(
            len(neg_query_result), replace=True)
    else:
        neg_query_result = neg_query_result.sample(
            len(pos_query_result), replace=True)
    
    pos_query_result["label"] = 1
    neg_query_result["label"] = 0

    # append, shuffle and set index
    output = pos_query_result.append(neg_query_result)
    output = output.dropna()
    output = output.sample(frac=1).reset_index(drop=True)
    output = output.set_index('id')

    with open("predict/predict_scripts/qbe_output/base_query_output.pickle", "wb+") as fp:
        pickle.dump(output, fp)

    # return both query results as one df with labels
    return output


def parse_gdb_df(df):

    query = ""
    curr_node = df[df["Node"]==0]
    left_child = df[df["ID"]==curr_node["Yes"].values[0]]
    right_child = df[df["ID"]==curr_node["No"].values[0]]
    numeric_feature = False

    while(curr_node["Feature"].values[0]!="Leaf"):

        # check for one hot encoded/categorical values
        if("_" in str(curr_node["Feature"].values[0])):
            feature_name = str(curr_node["Feature"].values[0]).split("_")[0]
            feature_value = "'" + "_".join(curr_node["Feature"].values[0].split("_")[1:]) + "'"
        # else for numerical values
        else:
            feature_name = curr_node["Feature"].values[0]
            feature_value = curr_node["Split"].values[0]   
            numeric_feature = True        
         # compare child nodes and extract query components
        if(left_child["Gain"].values[0]>right_child["Gain"].values[0]):
            curr_node = left_child
            left_child = df[df["ID"]==curr_node["Yes"].values[0]]
            right_child = df[df["ID"]==curr_node["No"].values[0]]
            if(numeric_feature):
                feature_comp = "<"
                numeric_feature = False
            else:
               feature_comp = "!="
        elif(right_child["Gain"].values[0]>left_child["Gain"].values[0]):
            curr_node = right_child
            left_child = df[df["ID"]==curr_node["Yes"].values[0]]
            right_child = df[df["ID"]==curr_node["No"].values[0]]
            if(numeric_feature):
                feature_comp = ">"
                numeric_feature = False
            else:
               feature_comp = "="

         # build query
        query = query +  " \t" + str(feature_name) + " " + str(feature_comp) + " " + str(feature_value) + "\n\t AND"

    return query

def gdb_to_sql(data):

    # group dataframe by tree
    data  = data.groupby(["Tree"])

    # list for storing sub-queries
    query_list = []

    # loop over every tree
    for name, group in data:

        # skip trees consisting out of only a leaf node
        if(len(set(group["Feature"]))==1):
            continue

        #call tree parsing function
        query = parse_gdb_df(group)        
        query = query.rsplit(' ', 1)[0]
        query = "(\n" + query + ")"
        query_list.append(query)

    query = "SELECT * FROM adult WHERE " + "\n OR ".join(query_list) + ";"
    return query

#  Generate SQL query conditions out of a tree by parsing its nodes
#  Code was provided by Denis Mayr Lima Martins and only sligtly changed
def get_predicate(tree, columns):
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = [columns[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    rules = []

    n_nodes = tree.tree_.node_count
    is_leaf = np.zeros(shape=n_nodes, dtype=bool)

    node_parenting = {}

    root = (0, [])
    stack = [root]

    # Trasverse tree to get an appropriate representation
    while len(stack) > 0:
        node_id, parents = stack.pop()
        node_parenting[node_id] = parents

        # If we have a test node
        if (left[node_id] != right[node_id]):
            l_heritage = parents[:]
            l_heritage.append(('l', node_id))

            r_heritage = parents[:]
            r_heritage.append(('r', node_id))

            stack.append((left[node_id], l_heritage))
            stack.append((right[node_id], r_heritage))
        else:
            is_leaf[node_id] = True

    # Get rules by finding paths from leaf to root
    for node, parents in node_parenting.items():
        if is_leaf[node] and value[node][0][1] > 0:
            current_rule = []
            for direction, p_node in parents:
                if direction == 'l':  # Left direction indicates that the parent condition is true
                    # check for one-hot-encoded column names
                    if(not "_" in features[p_node]):
                        current_rule.append(
                            features[p_node] + " <= " + str(threshold[p_node]))
                    else:
                        temp_string = features[p_node].split("_")
                        current_rule.append(
                            temp_string[0] + " != '" + temp_string[1] + "'")
                # Change the parent condition is parent direction is right (False)
                else:
                    # check for one-hot-encoded column names
                    if(not "_" in features[p_node]):
                        current_rule.append(
                            features[p_node] + " > " + str(threshold[p_node]))
                    else:
                        temp_string = features[p_node].split("_")
                        current_rule.append(
                            temp_string[0] + " = '" + "_".join(temp_string[1:]) + "'")
                current_rule.append('\n\t AND ')

            rules.append(''.join(current_rule[:-1]))  # Remove last AND
            rules.append('\n OR ')

    rules = rules[:-1]  # Remove last OR

    predicate = ''.join(rules)
    return predicate

# Gets a list of [SQL, QBE Set, Full Result Set, label col names, label row IDs] and a list of cols. Returns the same list but one hot encoded all columns in cols.


def one_hot_encode_list_items(x, cols):
    subset_column_headers = list(x[1].columns.values)
    existing_categorical_cols = set(subset_column_headers).intersection(cols)
    return [x[0], pd.get_dummies(x[1], columns=existing_categorical_cols), pd.get_dummies(x[2], columns=existing_categorical_cols)]

# turns specified cols in the dataset (passed as x) into vector representations using a word2vec model
# not used in final prototype, only in early experiments with QBE as regression 
def vectorize_list_items(x, cols, model, w2v_size):
    subset_column_headers = list(x[1].columns.values)
    existing_text_cols = set(subset_column_headers).intersection(cols)
    # iterate over every text column that should be vectorized
    for col in existing_text_cols:
        new_cols = []
        # generate new column names
        for index in range(0, w2v_size):
            new_col = col + "_wv" + str(index)
            new_cols.append(new_col)
        # fill new columns with the data from word2vec
        new_cols_data_1 = pd.DataFrame(model.wv[x[1][col]], columns=new_cols)
        new_cols_data_2 = pd.DataFrame(model.wv[x[2][col]], columns=new_cols)

        # copy index from the input to enable joining (inserting the new cols)
        new_cols_data_1 = new_cols_data_1.set_index(x[1].index)
        x[1] = x[1].join(new_cols_data_1)
        x[2] = x[2].join(new_cols_data_2)

        # drop columns that have now been replaced by word2vec data
        x[1] = x[1].drop(col, 1)
        x[2] = x[2].drop(col, 1)
        # sort columns
        x[1] = x[1].reindex(sorted(x[1].columns), axis=1)
        x[2] = x[2].reindex(sorted(x[2].columns), axis=1)

    return [x[0], x[1], x[2]]

# pad dataset with empty columns to make it the same size for every training iteration. Also sort columns to receive them always in the same order.
# not used in final prototype, only in early experiments with QBE as regression 
def pad_missing_cols(x, all_cols, vectorized_cols, w2v_size):

    # iterate over every text column that have been vectorized
    vec_cols = []
    for col in vectorized_cols:
        # generate new column names
        for index in range(0, w2v_size):
            vec_cols.append(col + "_wv" + str(index))

    existing_cols = list(x[1].columns.values)
    missing_cols = list(set(all_cols).union(
        set(vec_cols)).difference(existing_cols))

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
# not used in final prototype, only in early experiments with QBE as regression 
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
# not used in final prototype, only in early experiments with QBE as regression 
def clean_col_names(x):
    x[1].columns = x[1].columns.str.replace('<=', 'under')
    x[2].columns = x[2].columns.str.replace('<=', 'under')
    x[1].columns = x[1].columns.str.replace('>', 'over')
    x[2].columns = x[2].columns.str.replace('>', 'over')
    return x

# function to get subset of a table containing the most extreme points of each dimension
# not used in final prototype, only in early experiments with QBE as regression 
def get_min_max_sample(x, n=None, frac=None, random_state=5258):
    result_set = pd.DataFrame()
    for column in x.columns.values:
        x = x.sort_values(by=[column], ascending=True)
        result_set = result_set.append(x.head(1))
        result_set = result_set.append(x.tail(1))
    if(frac != None):
        result_set = result_set.append(
            x.sample(frac=frac, random_state=random_state))
    if(n != None):
        result_set = result_set.append(
            x.sample(n=n-len(result_set.index), random_state=random_state))
    return result_set
