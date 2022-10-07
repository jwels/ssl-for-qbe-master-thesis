from django.db import models
from django.contrib.postgres.fields import ArrayField

# Create your models here.
class Query(models.Model):
    sql_statement = models.CharField(max_length=600)

class TrainingData(models.Model):
    query = models.OneToOneField("sqlgenerator.Query", on_delete=models.CASCADE)   # ForeignKey("sqlgenerator.Query", on_delete=models.CASCADE, unique=True)
    result_cols = ArrayField(base_field=models.CharField(max_length=32, blank=True), blank=False)
    data_pickle_path = models.CharField(blank=False, max_length=255)    

class Adult(models.Model):
    age = models.IntegerField(blank=True, null=True)
    workclass = models.CharField(max_length=255, blank=True, null=True)
    fnlwgt = models.IntegerField(blank=True, null=True)
    education = models.CharField(max_length=255, blank=True, null=True)
    educationalnum = models.IntegerField(blank=True, null=True)
    maritalstatus = models.CharField(max_length=255, blank=True, null=True)
    occupation = models.CharField(max_length=255, blank=True, null=True)
    relationship = models.CharField(max_length=255, blank=True, null=True)
    race = models.CharField(max_length=255, blank=True, null=True)
    gender = models.CharField(max_length=255, blank=True, null=True)
    capitalgain = models.IntegerField(blank=True, null=True)
    capitalloss = models.IntegerField(blank=True, null=True)
    hoursperweek = models.IntegerField(blank=True, null=True)
    nativecountry = models.CharField(max_length=255, blank=True, null=True)
    income = models.CharField(max_length=255, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'adult'
