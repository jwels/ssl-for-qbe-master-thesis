from copy import deepcopy
from django import forms
from sqlgenerator.models import Adult

class UserColSelect(forms.Form):
    fields = Adult._meta.fields
    # fields = fields.remove("id")

    Columns = forms.MultipleChoiceField(widget=forms.CheckboxSelectMultiple, choices=[(c.name, c.name) for c in fields])