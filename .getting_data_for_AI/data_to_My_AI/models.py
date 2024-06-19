from django.db import models

class SourceModel(models.Model):
    uildverlegch = models.BigIntegerField()
    mark = models.BigIntegerField()
    motor_bagtaamj = models.BigIntegerField()
    xrop = models.BigIntegerField()
    joloo = models.BigIntegerField()
    uildverlesen_on = models.BigIntegerField()
    orj_irsen_on = models.BigIntegerField()
    hudulguur = models.BigIntegerField()
    hutlugch = models.BigIntegerField()
    yavsan_km = models.BigIntegerField()
    une = models.FloatField()

    def __str__(self):
        return f'{self.uildverlegch} {self.mark}'

class DestinationModel(models.Model):
    uildverlegch = models.BigIntegerField()
    mark = models.BigIntegerField()
    motor_bagtaamj = models.BigIntegerField()
    xrop = models.BigIntegerField()
    joloo = models.BigIntegerField()
    uildverlesen_on = models.BigIntegerField()
    orj_irsen_on = models.BigIntegerField()
    hudulguur = models.BigIntegerField()
    hutlugch = models.BigIntegerField()
    yavsan_km = models.BigIntegerField()
    une = models.FloatField()

    def __str__(self):
        return f'{self.uildverlegch} {self.mark}'
