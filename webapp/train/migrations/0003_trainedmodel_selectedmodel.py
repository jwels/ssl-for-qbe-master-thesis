# Generated by Django 4.0.5 on 2022-07-04 10:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('train', '0002_trainedmodel_createdat'),
    ]

    operations = [
        migrations.AddField(
            model_name='trainedmodel',
            name='selectedModel',
            field=models.BooleanField(default=False),
        ),
    ]
