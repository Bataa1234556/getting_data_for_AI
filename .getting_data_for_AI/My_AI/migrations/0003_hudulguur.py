# Generated by Django 5.0.6 on 2024-06-13 08:54

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('My_AI', '0002_alter_hutlugch_type_alter_mark_name_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='Hudulguur',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('type', models.CharField(choices=[('Бензин', 'Бензин'), ('Газ', 'Газ'), ('Дизель', 'Дизель'), ('Хайбрид', 'Хайбрид'), ('Цахилгаан', 'Цахилгаан')], max_length=100)),
            ],
        ),
    ]
