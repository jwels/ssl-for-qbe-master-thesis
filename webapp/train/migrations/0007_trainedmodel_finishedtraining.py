# Generated by Django 4.0.5 on 2022-07-04 16:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('train', '0006_alter_trainedmodel_modeltype'),
    ]

    operations = [
        migrations.AddField(
            model_name='trainedmodel',
            name='finishedTraining',
            field=models.BooleanField(default=False),
        ),
    ]
