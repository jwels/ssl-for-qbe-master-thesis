import pickle
from .models import Query, Adult, TrainingData
from django.db.models import Max, Min
from django.db import connections
import random
import pandas as pd
from pathlib import Path

# Function to generate SQL queries for a database to later use for training
def generate_queries(n, cols_min, cols_max):
    
    # get the cols/field names
    field_names = [f.name for f in Adult._meta.get_fields()]
    field_names.remove("id")
    field_types = [Adult._meta.get_field(f).get_internal_type() for f in field_names]
    where_ops_int_all = ["<", ">", "=", "BETWEEN", ">=", "<="]
    where_ops_char_all = ["=", "!="]
    where_conjunctions_all = ["AND", "AND", "OR"]
    # sort fields by data type
    integer_fields = []
    char_fields = []
    for index, entry in enumerate(field_types) :
        if entry == "IntegerField":
            integer_fields.append(field_names[index])
        if entry == "CharField":
            char_fields.append(field_names[index])
    # get the min and max values of the integer fields
    min_field_values = {str(f): list(Adult.objects.all().using("adult").aggregate(Min(f)).values())[0] for f in integer_fields}
    max_field_values = {str(f): list(Adult.objects.all().using("adult").aggregate(Max(f)).values())[0] for f in integer_fields}
    # get the set of possible values for the char fields (nested list)
    char_field_sets = {str(f): list(Adult.objects.order_by().using("adult").values_list(f, flat=True).distinct()) for f in char_fields}
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
        where_comp_int = [random.sample(range(min_field_values[entry], max_field_values[entry]), 1)[0] for entry in where_cols_int]
        where_comp_char = [random.sample(char_field_sets[entry], 1)[0] for entry in where_cols_char]
        
        # build the query string
        query = "SELECT * FROM adult WHERE " # Alternative to * if not all cols selected: |", ".join(sel_cols)|
        if(len(where_cols_int)>0):
            for j in range(0, len(where_cols_int)-2):
                if(where_ops_int[j]=="BETWEEN"):
                    where_comp_int[j] = str(where_comp_int[j]) + " AND " +  str(random.sample(range(where_comp_int[j], max_field_values[where_cols_int[j]]), 1)[0])
                query = query + where_cols_int[j] + " " + where_ops_int[j] + " " + str(where_comp_int[j]) + " " + random.sample(where_conjunctions_all, 1)[0] + " "
            if(where_ops_int[len(where_cols_int)-1]=="BETWEEN"):
                    where_comp_int[len(where_cols_int)-1] = str(where_comp_int[len(where_cols_int)-1]) + " AND " +  str(random.sample(range(where_comp_int[len(where_cols_int)-1], max_field_values[where_cols_int[len(where_cols_int)-1]]), 1)[0])
            query = query + where_cols_int[len(where_cols_int)-1] + " " + where_ops_int[len(where_cols_int)-1] + " " + str(where_comp_int[len(where_cols_int)-1])
            if(len(where_cols_char)==0):
                query = query + ";"
            else:
                query = query + " " + random.sample(where_conjunctions_all, 1)[0] + " "
        if(len(where_cols_char)>0):
            for j in range(0, len(where_cols_char)-2):
                query = query + where_cols_char[j] + " " + where_ops_char[j] + " '" + where_comp_char[j] + "' " + random.sample(where_conjunctions_all, 1)[0] + " "
            query = query + where_cols_char[len(where_cols_char)-1] + " " + where_ops_char[len(where_cols_char)-1] + " '"+ where_comp_char[len(where_cols_char)-1] + "';"

        new_query = Query(sql_statement=query)
        new_query.save()
    return True
