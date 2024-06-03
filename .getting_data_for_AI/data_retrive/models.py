from django.db import models

class CarData(models.Model):
    manufacturer = models.CharField(max_length=100, null=True)
    model = models.CharField(max_length=100, null=True)
    posted_date = models.DateField(null=True)
    engine_capacity = models.CharField(max_length=100, null=True)
    transmission = models.CharField(max_length=100, null=True)
    steering = models.CharField(max_length=100, null=True)
    type = models.CharField(max_length=100, null=True)
    color = models.CharField(max_length=100, null=True)
    manufacture_year = models.IntegerField(null=True)
    import_year = models.IntegerField(null=True)
    drive = models.CharField(max_length=100, null=True)
    interior_color = models.CharField(max_length=100, null=True)
    lease = models.CharField(max_length=100, null=True)
    address = models.CharField(max_length=100, null=True, default="")
    drive_type = models.CharField(max_length=100, null=True)
    mileage = models.CharField(max_length=100, null=True)
    license_plate = models.CharField(max_length=100, null=True)
    doors = models.CharField(max_length=100, null=True)
    price = models.IntegerField(null=True)
    unique_id = models.CharField(max_length=100, null=True)

    def __str__(self):
        return f"{self.manufacturer} {self.model}"
