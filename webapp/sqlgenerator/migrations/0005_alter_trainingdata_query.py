# Generated by Django 4.0.5 on 2022-07-05 13:26

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('sqlgenerator', '0004_trainingdata'),
    ]

    operations = [
        migrations.AlterField(
            model_name='trainingdata',
            name='query',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='sqlgenerator.query', unique=True),
        ),
    ]
