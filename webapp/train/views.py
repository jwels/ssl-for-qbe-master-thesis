from django.shortcuts import render

# Create your views here.

from django.http import HttpResponse
from django.template import loader
from django.shortcuts import redirect

from .forms import TrainedModelForm
from .models import TrainedModel
import subprocess, json

def index(request):
    all_models = TrainedModel.objects.all().order_by('id')
    selected_model = TrainedModel.objects.filter(selectedModel=True)
    if(len(selected_model)>0):
        selected_model = selected_model[0]
    
    template = loader.get_template('train/index.html')
    context = {
        'models': all_models,
        'selectedModel': selected_model
    }

    return HttpResponse(template.render(context, request))

def dropModel(request):
    TrainedModel.objects.filter(id=request.POST["modelId"]).delete()
    return redirect(index)

def selectModel(request):
    TrainedModel.objects.filter(selectedModel=True).update(selectedModel=False)
    TrainedModel.objects.filter(id=request.POST["modelId"]).update(selectedModel=True)
    return redirect(index)

def newModel(request):
    form = TrainedModelForm()
    return render(request, 'train/newModel.html', {'form': form})

def createNewModel(request):
    new_model = TrainedModelForm(request.POST)
    new_model = new_model.save()
    subprocess.Popen(["python3", "manage.py", "startTuning", str(new_model.id)], stdin=None, stdout=None, stderr=None)
    return redirect(index)

def debugView(request):
    # just used to easily change something in the database for debugging
    print("changing model")
    model = TrainedModel.objects.filter(id=290)[0]
    model.modelParams = json.loads('{"max_depth": 2.0, "min_samples_leaf": 1.0, "min_samples_split": 2.0}')
    model.save()
    return redirect(index)