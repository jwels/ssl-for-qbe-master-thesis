# Generated by Django 4.0.5 on 2022-07-05 12:49

import django.contrib.postgres.fields
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('sqlgenerator', '0003_alter_query_sql_statement'),
    ]

    operations = [
        migrations.CreateModel(
            name='TrainingData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('result_cols', django.contrib.postgres.fields.ArrayField(base_field=models.CharField(blank=True, max_length=32), size=None)),
                ('data_pickle_path', models.CharField(max_length=255)),
                ('query', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='sqlgenerator.query')),
            ],
        ),
    ]
