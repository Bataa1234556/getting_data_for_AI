from .models import SourceModel, DestinationModel

def transfer_data():
    source_data = SourceModel.objects.all()
    for item in source_data:
        DestinationModel.objects.create(
            uildverlegch=item.uildverlegch,
            mark=item.mark,
            motor_bagtaamj=item.motor_bagtaamj,
            xrop=item.xrop,
            joloo=item.joloo,
            uildverlesen_on=item.uildverlesen_on,
            orj_irsen_on=item.orj_irsen_on,
            hudulguur=item.hudulguur,
            hutlugch=item.hutlugch,
            yavsan_km=item.yavsan_km,
            une=item.une
        )
