# Generated by Django 4.0.5 on 2022-07-04 12:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('train', '0005_trainedmodel_modeltype'),
    ]

    operations = [
        migrations.AlterField(
            model_name='trainedmodel',
            name='modelType',
            field=models.CharField(choices=[('gdb', 'Gradient Boosting'), ('dt', 'Decision Tree'), ('test', 'Test/no real model')], max_length=255),
        ),
    ]
