from django.db import models

# Create your models here.

class TrainedModel(models.Model):
    name = models.CharField(max_length=255, blank=False, null=False)
    modelType = models.CharField(max_length=255, blank=False, choices=[("gdbss", "Gradient Boosting Semi-Supervised"), ("gdbs", "Gradient Boosting Supervised"), ("dtss", "Decision Tree Semi-Supervised"), ("dt", "Decision Tree Supervised"), ("rfss", "Random Forest Semi-Supervised"), ("rf", "Random Forest Supervised")], null=False)
    modelDescr = models.CharField(max_length=255, blank=True, null=True)
    createdAt = models.DateTimeField(auto_now_add=True, blank=False)
    selectedModel = models.BooleanField(default=False)
    finishedTraining = models.BooleanField(default=False)
    f1scoreTest = models.FloatField(null=True)
    f1scoreVal = models.FloatField(null=True)
    modelParams = models.JSONField(null=True)
    maxEvals = models.IntegerField(null=False)
    removeLowestCertaintyPercentage = models.FloatField(null=True, default=0.0)
