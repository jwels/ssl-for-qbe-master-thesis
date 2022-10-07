from django.core.management.base import BaseCommand
from sqlgenerator.models import Query, TrainingData, Adult
from sklearn.preprocessing import MinMaxScaler
from django.db import connections
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sqlgenerator.helper_functions import build_and_run_base_query

class Command(BaseCommand):
    help = 'Start process to generate training data from the prev. generated SQL Queries.'

    def add_arguments(self, parser):
        parser.add_argument('db_names', nargs='+', type=str)

    # Function to generate training data from the SQL queries generated previously
    def handle(self, *args, **options):
        for db_name in options['db_names']:
      
            # get list of sql queries
            sql_list = Query.objects.all()

            # Iterate over list of SQL queries
            for query in sql_list:

                #skip if query was already used in previous run
                if TrainingData.objects.filter(query=query).exists():
                        continue
                # empty list of all results
                dataset = []
                # Run Query
                result_fullset_pos = pd.read_sql(query.sql_statement, connections[db_name])
                # column names that the user wanted
                result_cols = list(result_fullset_pos.columns.values)   
                # get negative examples 
                result_fullset_negatives = pd.read_sql("SELECT " + ', '.join(result_cols) + " FROM " + db_name +" EXCEPT " + query.sql_statement, connections[db_name])
                # Get random subset of results to use as QBE input if total result set is big enough
                if(len(result_fullset_pos)<5 or len(result_fullset_negatives)<5):
                    query.delete()
                    continue
                result_subset_pos = result_fullset_pos.sample(n=5, random_state=5258) 
                # get max number to sample
                if(len(result_fullset_negatives)<5):
                    sample_size = len(result_fullset_negatives)
                else:
                    sample_size = 5
                # sample negative examples from all negatives
                result_subset_negatives = result_fullset_negatives.sample(n=sample_size, random_state=5258)

                # use input extension mechanism (construct a base query to get examples from the DB that match input)
                result_subset_pos["label"] = 1
                result_subset_negatives["label"] = 0
                base_query_output = build_and_run_base_query(result_subset_pos.append(result_subset_negatives))  
                result_subset_pos = base_query_output[base_query_output['label']==1]
                result_subset_negatives = base_query_output[base_query_output['label']==0]
                result_subset_pos = result_subset_pos.drop(columns=["label"])
                result_subset_negatives = result_subset_negatives.drop(columns=["label"])   
                # Delete Query if pos. or negative set is too small
                if(len(result_subset_pos)<5 or len(result_subset_negatives)<5):
                    query.delete()
                    continue

                # get all existing categorical fields
                field_names = [f.name for f in Adult._meta.get_fields()]
                field_names.remove("id")
                char_fields = []
                # numeric_fields = []
                field_types = [Adult._meta.get_field(f).get_internal_type() for f in field_names]
                for index, entry in enumerate(field_types):
                    # if entry == "IntegerField":
                    #     numeric_fields.append(field_names[index])
                    if entry == "CharField":
                        char_fields.append(field_names[index])                        
 
                # set positive label
                result_subset_pos["label"] = 1
                result_fullset_pos["label"] = 1
  
                # set negative label
                result_subset_negatives["label"] = 0
                result_fullset_negatives["label"] = 0              
                # append negatives to full result set and encode the categorical cols
                qbe_resultset = result_fullset_pos.append(result_fullset_negatives)                
                qbe_resultset = pd.get_dummies(qbe_resultset, columns=char_fields)
                # append negatives to QBE set and encode the categorical cols
                qbe_subset = result_subset_pos.append(result_subset_negatives)
                qbe_subset = pd.get_dummies(qbe_subset, columns=char_fields)
                # add potentially missing cols after one hot encoding to smaller set (due to values not occurring)
                missing_cols = set( qbe_resultset.columns ) - set( qbe_subset.columns )
                for column in missing_cols:
                    qbe_subset[column] = 0
                # drop NAs
                qbe_subset = qbe_subset.dropna()
                qbe_resultset = qbe_resultset.dropna()
                dataset.append([qbe_subset, qbe_resultset])
    
                # store results
                Path("data/ds-generator/"+ db_name).mkdir(parents=True, exist_ok=True)
                file_path = "data/ds-generator/"+ db_name + "/sql_query_" + str(query.id) +"_data.pickle"
                with open(file_path, "wb+") as fp:
                    pickle.dump(dataset, fp)
                training_data = TrainingData(query=query, result_cols=result_cols, data_pickle_path=file_path)
                training_data.save()
            