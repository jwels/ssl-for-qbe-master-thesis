# Generated by Django 4.0.5 on 2022-06-29 13:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('sqlgenerator', '0002_adult'),
    ]

    operations = [
        migrations.AlterField(
            model_name='query',
            name='sql_statement',
            field=models.CharField(max_length=600),
        ),
    ]