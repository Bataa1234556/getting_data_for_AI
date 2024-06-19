from django.core.management.base import BaseCommand
from data_to_My_AI.models import SourceModel, DestinationModel

class Command(BaseCommand):
    help = 'Remove duplicates from DestinationModel based on SourceModel'

    def handle(self, *args, **kwargs):
        # Logic to remove duplicates
        self.remove_duplicates()
        self.stdout.write(self.style.SUCCESS('Duplicates removed successfully!'))

    def remove_duplicates(self):
        source_objects = SourceModel.objects.all()

        for source_obj in source_objects:
            duplicate_objs = DestinationModel.objects.filter(
                uildverlegch=source_obj.uildverlegch,
                mark=source_obj.mark,
                motor_bagtaamj=source_obj.motor_bagtaamj,
                xrop=source_obj.xrop,
                joloo=source_obj.joloo,
                uildverlesen_on=source_obj.uildverlesen_on,
                orj_irsen_on=source_obj.orj_irsen_on,
                hudulguur=source_obj.hudulguur,
                hutlugch=source_obj.hutlugch,
                yavsan_km=source_obj.yavsan_km,
                une=source_obj.une
            )

            if duplicate_objs.exists():
                duplicate_objs.delete()

        # Check if there are no more duplicates left
        remaining_duplicates = DestinationModel.objects.filter(
            uildverlegch__in=SourceModel.objects.values_list('uildverlegch', flat=True),
            mark__in=SourceModel.objects.values_list('mark', flat=True),
            motor_bagtaamj__in=SourceModel.objects.values_list('motor_bagtaamj', flat=True),
            xrop__in=SourceModel.objects.values_list('xrop', flat=True),
            joloo__in=SourceModel.objects.values_list('joloo', flat=True),
            uildverlesen_on__in=SourceModel.objects.values_list('uildverlesen_on', flat=True),
            orj_irsen_on__in=SourceModel.objects.values_list('orj_irsen_on', flat=True),
            hudulguur__in=SourceModel.objects.values_list('hudulguur', flat=True),
            hutlugch__in=SourceModel.objects.values_list('hutlugch', flat=True),
            yavsan_km__in=SourceModel.objects.values_list('yavsan_km', flat=True),
            une__in=SourceModel.objects.values_list('une', flat=True)
        )

        if not remaining_duplicates.exists():
            self.stdout.write(self.style.SUCCESS('Successfully removed all duplicates!'))
            return
