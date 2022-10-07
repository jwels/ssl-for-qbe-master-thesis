from django.http import HttpResponse
from django.template import loader
from django.shortcuts import redirect
import os
from .models import Query, Adult, TrainingData
from .services import generate_queries
import subprocess

def index(request):
    queries = Query.objects.all().order_by('id')
    n_of_trainingdata = TrainingData.objects.all().count()
    template = loader.get_template('sqlgenerator/index.html')
    n_of_fields = len([f.name for f in Adult._meta.get_fields()])-1 #substract id column
    context = {
        'queries': queries,
        'n_of_fields': n_of_fields,
        'n_of_trainingdata': n_of_trainingdata
    }
    return HttpResponse(template.render(context, request))

def generate_queries_view(request):
    generate_queries(int(request.POST["nOfQueries"]), int(request.POST["minCols"]), int(request.POST["maxCols"]))
    return redirect(index)

def generate_training_data_view(request):
    #start dataset generation sub process. Runs SQL Queries on database to generate training data
    subprocess.Popen(["python3", "manage.py", "generate_classif_training_data", "adult"], stdin=None, stdout=None, stderr=None)
    return redirect(index)

def drop_all(request):
    Query.objects.all().delete()
    return redirect(index)

def drop_all_trainingdata(request):
    path_list = list(TrainingData.objects.all())
    path_list = [entry.data_pickle_path for entry in path_list]
    for entry in path_list:
        os.remove(entry)
    TrainingData.objects.all().delete()
    return redirect(index)