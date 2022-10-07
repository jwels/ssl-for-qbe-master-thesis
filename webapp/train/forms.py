from django import forms
from django.forms import ModelForm
from .models import TrainedModel

class TrainedModelForm(ModelForm):
    name = forms.CharField(label='Name', initial="Some Model")
    modelType = forms.ChoiceField(choices=TrainedModel.modelType.field.choices ,label='Type')
    maxEvals = forms.IntegerField(label='Max. Number of Iterations', initial=1, min_value=0, max_value=999999)
    removeLowestCertaintyPercentage = forms.DecimalField(label = "Ignore pseudo-labels w. lowest certainty (fraction)", initial=0.0, min_value=0.0, max_value=1.0)
    modelDescr = forms.CharField(label='Description')

    class Meta:
      model = TrainedModel
      fields = ['name', 'modelType', 'maxEvals', 'removeLowestCertaintyPercentage', 'modelDescr']