# Generated by Django 4.0.5 on 2022-07-11 18:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('train', '0010_rename_f1score_trainedmodel_f1scoretest_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='trainedmodel',
            name='modelType',
            field=models.CharField(choices=[('gdbss', 'Gradient Boosting Semi-Supervised'), ('gdbr', 'Gradient Boosting Regression'), ('gdbs', 'Gradient Boosting Supervised'), ('dt', 'Decision Tree'), ('test', 'Test/no real model')], max_length=255),
        ),
    ]