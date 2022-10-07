# Generated by Django 4.0.5 on 2022-07-05 12:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('train', '0007_trainedmodel_finishedtraining'),
    ]

    operations = [
        migrations.AlterField(
            model_name='trainedmodel',
            name='modelType',
            field=models.CharField(choices=[('gdbr', 'Gradient Boosting Regression'), ('gdbss', 'Gradient Boosting Semi-Supervised'), ('gdbs', 'Gradient Boosting Supervised'), ('dt', 'Decision Tree'), ('test', 'Test/no real model')], max_length=255),
        ),
    ]