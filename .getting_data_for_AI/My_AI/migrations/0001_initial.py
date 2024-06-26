# Generated by Django 5.0.6 on 2024-06-13 08:00

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Joloo',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('type', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='Mark',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='OrjIrsenOn',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('year', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='Uildverlegch',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='MotorBagtaamj',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('size', models.CharField(max_length=100)),
                ('mark', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='My_AI.mark')),
            ],
        ),
        migrations.CreateModel(
            name='Hutlugch',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('type', models.CharField(max_length=100)),
                ('orj_irsen_on', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='My_AI.orjirsenon')),
            ],
        ),
        migrations.AddField(
            model_name='mark',
            name='uildverlegch',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='My_AI.uildverlegch'),
        ),
        migrations.CreateModel(
            name='UildverlesenOn',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('year', models.CharField(max_length=100)),
                ('joloo', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='My_AI.joloo')),
            ],
        ),
        migrations.AddField(
            model_name='orjirsenon',
            name='uildverlesen_on',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='My_AI.uildverlesenon'),
        ),
        migrations.CreateModel(
            name='Xrop',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('type', models.CharField(max_length=100)),
                ('motor_bagtaamj', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='My_AI.motorbagtaamj')),
            ],
        ),
        migrations.AddField(
            model_name='joloo',
            name='xrop',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='My_AI.xrop'),
        ),
        migrations.CreateModel(
            name='YavsanKm',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('distance', models.CharField(max_length=100)),
                ('hutlugch', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='My_AI.hutlugch')),
            ],
        ),
    ]
