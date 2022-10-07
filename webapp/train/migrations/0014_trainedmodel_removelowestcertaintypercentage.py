# Generated by Django 4.0.5 on 2022-07-28 09:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('train', '0013_trainedmodel_maxevals'),
    ]

    operations = [
        migrations.AddField(
            model_name='trainedmodel',
            name='removeLowestCertaintyPercentage',
            field=models.FloatField(default=0.0, null=True),
        ),
    ]
