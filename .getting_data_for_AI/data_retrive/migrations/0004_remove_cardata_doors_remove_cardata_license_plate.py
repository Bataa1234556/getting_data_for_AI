# Generated by Django 5.0.6 on 2024-06-04 05:58

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('data_retrive', '0003_remove_cardata_address_remove_cardata_lease'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='cardata',
            name='doors',
        ),
        migrations.RemoveField(
            model_name='cardata',
            name='license_plate',
        ),
    ]
