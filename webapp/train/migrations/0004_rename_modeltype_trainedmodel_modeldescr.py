# Generated by Django 4.0.5 on 2022-07-04 12:23

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('train', '0003_trainedmodel_selectedmodel'),
    ]

    operations = [
        migrations.RenameField(
            model_name='trainedmodel',
            old_name='modelType',
            new_name='modelDescr',
        ),
    ]
