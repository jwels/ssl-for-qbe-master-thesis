import random, os
import pandas as pd
from pathlib import Path

# dataset = pd.read_csv(
#     Path.joinpath(Path(Path.cwd()).resolve().parents[0], str(os.getcwd()), 'data', 'raw', 'stars.csv'), 
#     na_values="?").reset_index(drop=True)
# dataset = dataset.drop(columns=["objID"])
# dataset = dataset.drop(columns=["rerunID"])
# dataset = dataset.drop(columns=["specobjID"])

dataset = pd.read_csv(
    Path.joinpath(Path(Path.cwd()).resolve().parents[0], str(os.getcwd()), 'data', 'raw', 'movies.csv'), 
    na_values="?", sep="\t").reset_index(drop=True)

# parameters
cols_min = 3
cols_max = 4
n = 100
# sdss field_names ["alpha", "delta", "u", "g", "r", "i", "z", "runID", "camcol", "fieldID", "objclass", "redshift", "plate", "MJD", "fiberID"]
field_names = ["title", "production_year", "genre", "language", "lead_actor", "lead_gender", "lead_country", "lead_birthyear", "lead_age", "lead_height"]
field_types = ["categoric", "numeric", "categoric", "categoric", "categoric", "categoric", "categoric", "numeric", "categoric", "categoric"]

where_ops_int_all = ["<", ">", "==", "BETWEEN", ">=", "<="]
where_ops_char_all = ["==", "!="]
where_conjunctions_all = ["AND", "AND", "OR"]
# sort fields by data type
integer_fields = []
char_fields = []
for index, entry in enumerate(field_types) :
    if entry == "numeric":
        integer_fields.append(field_names[index])
    if entry == "categoric":
        char_fields.append(field_names[index])#
# get the min and max values of the integer fields
min_field_values = {str(f): dataset[f].min() for f in integer_fields}
max_field_values = {str(f): dataset[f].max() for f in integer_fields}
print(min_field_values)
print(max_field_values)
# get the set of possible values for the char fields (nested list)
char_field_sets = {str(f): set(dataset[f]) for f in char_fields}

for i in range(0,n):
    # randomly choose an amount of cols to draw from total list
    n_cols_select = random.randint(cols_min, cols_max)
    n_cols_where = random.randint(cols_min, cols_max)
    int_char_margin = random.randint(0, n_cols_where)
    # draw as often from the lists        
    sel_cols = random.sample(field_names, n_cols_select)
        # Case 1: enough integer fields and char fields exist
    if(len(integer_fields)>int_char_margin and len(char_fields)>n_cols_where-int_char_margin):
        where_cols_int = random.sample(integer_fields, int_char_margin)
        where_cols_char = random.sample(char_fields, n_cols_where-int_char_margin)
        # Case 2: enough integer fields but not enough char fields exist
    elif(len(integer_fields)>int_char_margin and not len(char_fields)>n_cols_where-int_char_margin):
        where_cols_char = random.sample(char_fields, len(char_fields))
        where_cols_int = random.sample(integer_fields, n_cols_where-len(where_cols_char))
        # where_cols_int.append(random.sample(integer_fields, n_cols_where-(int_char_margin+len(where_cols_char)))[0])
        # Case 3: not enough integer fields but enough char fields exist
    else:
        where_cols_int = random.sample(integer_fields, len(integer_fields))
        where_cols_char = random.sample(char_fields, n_cols_where-len(where_cols_int))
    # draw random operators for where clause comparison
    where_ops_int = random.choices(where_ops_int_all, k=len(where_cols_int))
    where_ops_char = random.choices(where_ops_char_all, k=len(where_cols_char))
    # draw random values to compare to
    where_comp_int = [random.sample(range(int(min_field_values[entry]), int(max_field_values[entry])), 1)[0] for entry in where_cols_int]
    where_comp_char = [random.sample(char_field_sets[entry], 1)[0] for entry in where_cols_char]
    
    # build the query string
    query = ""
    if(len(where_cols_int)>0):
        for j in range(0, len(where_cols_int)-2):
            if(where_ops_int[j]=="BETWEEN"):
                where_comp_int[j] = str(where_comp_int[j]) + " AND " +  str(random.sample(range(where_comp_int[j], max_field_values[where_cols_int[j]]), 1)[0])
            query = query + where_cols_int[j] + " " + where_ops_int[j] + " " + str(where_comp_int[j]) + " " + random.sample(where_conjunctions_all, 1)[0] + " "
        if(where_ops_int[len(where_cols_int)-1]=="BETWEEN"):
                where_comp_int[len(where_cols_int)-1] = str(where_comp_int[len(where_cols_int)-1]) + " AND " +  str(random.sample(range(int(where_comp_int[len(where_cols_int)-1]), int(max_field_values[where_cols_int[len(where_cols_int)-1]])), 1)[0])
        query = query + where_cols_int[len(where_cols_int)-1] + " " + where_ops_int[len(where_cols_int)-1] + " " + str(where_comp_int[len(where_cols_int)-1])
        if(len(where_cols_char)>0):
            query = query + " " + random.sample(where_conjunctions_all, 1)[0] + " "
    if(len(where_cols_char)>0):
        for j in range(0, len(where_cols_char)-2):
            query = query + where_cols_char[j] + " " + where_ops_char[j] + ' "' + where_comp_char[j] + '" ' + random.sample(where_conjunctions_all, 1)[0] + " "
        query = query + where_cols_char[len(where_cols_char)-1] + " " + where_ops_char[len(where_cols_char)-1] + ' "'+ where_comp_char[len(where_cols_char)-1] + '"'

    query = query.replace("AND", " and")
    query = query.replace("OR", " or")
    if(not("BETWEEN" in query) and not("'" in query) and len(dataset.query(query))==0):
        n = n + 1
        continue
    query = query.replace("_", "")
    with open('data/query_generator_output.csv','a') as f:
        f.write(query + "\n")