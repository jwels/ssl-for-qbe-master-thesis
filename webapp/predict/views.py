from django.http import HttpResponse
from django.shortcuts import redirect
from django.template import loader
from train.models import TrainedModel
from sqlgenerator.models import Adult
import pandas as pd
import pickle
import subprocess
from .forms import UserColSelect
import os, re
from django.http import JsonResponse

def index(request):
    selected_model = TrainedModel.objects.filter(selectedModel=True)
    if(len(selected_model)>0):
        selected_model = selected_model[0]
    field_names = [f.name for f in Adult._meta.get_fields()]
    field_names.remove("id")
    max_n_of_entries = Adult.objects.all().using("adult").count()
    template = loader.get_template('predict/index.html')
    # read predictions from file
    if(os.path.isfile("predict/predict_scripts/qbe_output/qbe_output_df.pickle")):
        with open("predict/predict_scripts/qbe_output/qbe_output_df.pickle", "rb") as fp:
                result_set = pickle.load(fp)
        col_list = [col.split("_")[0] if ("_" in col) else col for col in result_set.columns]
        if request.method == 'POST':
            form = UserColSelect(request.POST)
            if form.is_valid():
                user_selected_cols = form.cleaned_data.get('Columns')
                for col in user_selected_cols:
                    col_list.append(col)
            else:
                user_selected_cols = []
        else:
            form = UserColSelect
            user_selected_cols = []

        result_set = pd.DataFrame(Adult.objects.all().filter(id__in=result_set.index).values(*col_list).using("adult"))
    else:
        result_set = []
        col_list = []
        form = UserColSelect
        user_selected_cols = []
    # read base Query from file
    if(os.path.isfile("predict/predict_scripts/qbe_output/base_query_pos.pickle")):
        with open("predict/predict_scripts/qbe_output/base_query_pos.pickle", "rb") as fp:
                base_query = pickle.load(fp)
    else:
        base_query = "No query found yet."
    # read modeled Query from file
    if(os.path.isfile("predict/predict_scripts/qbe_output/qbe_output_query.pickle")):
        with open("predict/predict_scripts/qbe_output/qbe_output_query.pickle", "rb") as fp:
                query = pickle.load(fp)
    else:
        query = "No query found yet."
    context = {
        'selectedModel': selected_model,
        'existing_cols': field_names,
        'n_rows': 4,
        'n_cols': 2,
        'min_rows': 4,
        'min_cols': 1,
        'max_rows': 50,
        'max_cols': 3, #,len(field_names),
        'predictions': result_set,
        'max_number_entries': max_n_of_entries,
        'form': form,
        'found_query': query,
        'base_query': base_query
    }
    return HttpResponse(template.render(context, request))

def runQuery(request):    
    selected_model = TrainedModel.objects.filter(selectedModel=True)
    if(len(selected_model)>0):
        selected_model = selected_model[0]
    colnames = []
    labels = []
    cells = []
    print(request.POST)
    for key, value in request.POST.items():   # iter on both keys and values
        if key.startswith('select'):
                colnames.append(value)
                continue
        if key.startswith('checkbox'):
            if value == "on":
                labels.append(1)
            elif value == "off":
                labels.append(0)
            continue
        if key.startswith("cell_"):
            cells.append(value)
    data = pd.DataFrame(columns=colnames)
    row_data = []
    for i, entry in enumerate(cells):
        row_data.append(entry)
        if((i+1) % int(request.POST["n_cols"]) == 0):
            data = data.append(pd.Series(row_data,index=colnames),ignore_index=True )
            row_data = []
    data["label"] = labels
    print(data)
    with open("predict/predict_scripts/qbe_input/qbe_input_df.pickle", "wb+") as fp:
        pickle.dump(data, fp)
    prediction_process = subprocess.Popen(["python3", "manage.py", "predict", str(selected_model.id)], stdin=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = prediction_process.communicate()
    print(output)
    return redirect(index)

def runExampleQuery(request):
    selected_model = TrainedModel.objects.filter(selectedModel=True)
    if(len(selected_model)>0):
        selected_model = selected_model[0]
    # get correct example by id
    if(request.POST["exampleQueryId"] == "1"):
        data = pd.DataFrame({
        "age":[51, 56, 71, 89, 61, 59, 55, 23, 50, 55, 71, 25, 42, 67],
        "nativecountry":["Mexico", "Mexico", "Mexico", "Mexico", "Mexico", "Mexico", "United-States", "United-States", "Mexico", "Germany", "Japan", "Mexico", "United-States", "United-States"],
        "label": [1,1,1,1,1,1,0,0,0,0,0,0,0,0]
        })
    if(request.POST["exampleQueryId"] == "2"):
        data = pd.DataFrame({
        "age":[30, 50, 20, 99, 20, 99],
        "gender":["Female", "Female", "Female", "Female", "Male", "Male"], #, "Female", "Female", "Female", "Female", "Female", "Female", "Female", "Female", "Female", "Female", "Female"],        
        "label": [1,1,0,0,0,0]
        })
    if(request.POST["exampleQueryId"] == "3"):
        data = pd.DataFrame({
        "age":[20,40,41,99,20,99,20,99,20,99],
        "gender":["Male","Male","Male","Male","Female","Female","Male","Male","Male","Male"], 
        "nativecountry":["Germany", "Germany","Germany","Germany","Germany","Germany","United-States","United-States","Mexico","Mexico"],       
        "label": [1,1,0,0,0,0,0,0,0,0]
        })
    if(request.POST["exampleQueryId"] == "4"):
        data = pd.DataFrame({
        "age":["[40,90]","[17,40]","[17,40]","[40,90]","[17,90]","[17,90]","[17,90]"],
        "occupation":["Exec-managerial","Sales","Exec-managerial","Sales","Prof-specialty","Craft-repair","Farming-fishing"], 
        "label": [1,1,0,0,0,0,0]
        })
    print(data)
    with open("predict/predict_scripts/qbe_input/qbe_input_df.pickle", "wb+") as fp:
        pickle.dump(data, fp)
    prediction_process = subprocess.Popen(["python3", "manage.py", "predict", str(selected_model.id)], stdin=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = prediction_process.communicate()
    print(output)
    return redirect(index)


# return auto complete suggesions to the frontend
def getAutoCompletes(request):

    column_query_string = str(request.GET["column"]+"__startswith")    
    if(bool(re.match("\[{1}", request.GET["q"]))):
        result =  getSyntaxAutoComplete(request.GET["column"], request.GET["q"])
        response = JsonResponse(list(result), safe=False)
        return response
    else:  
        filter_condition = {column_query_string: str(request.GET["q"])}
        list_unique = Adult.objects.all().using("adult").filter(**filter_condition).values_list(request.GET["column"], flat=True).distinct()
        response = JsonResponse(list(list_unique), safe=False)
        return response 

# create autocomplete suggestions if the range syntax [x,y] was typed by the user
def getSyntaxAutoComplete(column, query):
    # remove_bracket = re.sub("\[{1}", "", request.GET["q"])
    # remove_bracket = re.sub("\\d*,", "", remove_bracket)
    # Case 1: just a first bracket typed
    if(re.match("\[{1}$", str(query))):
        list_unique = Adult.objects.all().using("adult").values_list(column, flat=True).distinct().order_by(column)
        res_list = []
        for entry in list_unique:
            res_list.append("["+str(entry)+",")
        return res_list
    # Case 2: first bracket + number typed
    elif(re.match("\[{1}\d*$", str(query))):
        # read out the values in brackets [x,y]
        values = re.search('\[{1}(.*)', query, re.IGNORECASE)
        value_min = values.group(1)
        column_query_string = str(column+"__startswith")
        filter_condition = {column_query_string: str(value_min)}
        list_unique = Adult.objects.all().using("adult").filter(**filter_condition).values_list(column, flat=True).distinct().order_by(column)
        res_list = []
        for entry in list_unique:
            res_list.append("["+str(entry)+",")
        return res_list
    # Case 3: typing second number
    elif(re.match("\[{1}\d*,\s*\d*", str(query))):
        # read out the values in brackets [x,y]
        values = re.search('\[{1}(.*),\s*(.*)', query, re.IGNORECASE)
        value_min = values.group(1)
        value_max = values.group(2)
        column_query_string_1 = str(column+"__gt")
        column_query_string_2 = str(column+"__startswith")
        filter_condition_1 = {column_query_string_1: str(value_min)}
        filter_condition_2 = {column_query_string_2: str(value_max)}
        list_unique = Adult.objects.all().using("adult").filter(**filter_condition_1, **filter_condition_2).values_list(column, flat=True).distinct().order_by(column)
        res_list = []
        for entry in list_unique:
            res_list.append("["+str(value_min)+","+str(entry)+"]")
        return res_list
  