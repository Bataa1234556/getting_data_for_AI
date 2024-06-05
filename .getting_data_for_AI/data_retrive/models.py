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
    drive_type = models.CharField(max_length=100, null=True)
    mileage = models.CharField(max_length=100, null=True)
    price = models.IntegerField(null=True)
    unique_id = models.CharField(max_length=100, null=True)

    def __str__(self):
        return f"Manufacturer: {self.manufacturer}, Model: {self.model}, Posted Date: {self.posted_date}, Engine Capacity: {self.engine_capacity}, Transmission: {self.transmission}, Steering: {self.steering}, Type: {self.type}, Color: {self.color}, Manufacture Year: {self.manufacture_year}, Import Year: {self.import_year}, Drive: {self.drive}, Interior Color: {self.interior_color}, Drive Type: {self.drive_type}, Mileage: {self.mileage}, Price: {self.price}, Unique ID: {self.unique_id}"

