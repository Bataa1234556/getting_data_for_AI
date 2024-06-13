from django import forms
from .models import Uildverlegch, Mark, MotorBagtaamj, Xrop, Joloo, UildverlesenOn, OrjIrsenOn, Hutlugch, Hudulguur, YavsanKm

class SelectionForm(forms.Form):
    uildverlegch = forms.ChoiceField(choices=Uildverlegch.UILVERLEGCH_CHOICES, required=True)
    hudulguur = forms.ChoiceField(choices=Hudulguur.HUDULGUUR_CHOICES, required=True)
    mark = forms.ChoiceField(choices=Mark.MARK_CHOICES, required=True)
    motor_bagtaamj = forms.CharField(required=True)
    xrop = forms.ChoiceField(choices=Xrop.XROP_CHOICES, required=True)
    joloo = forms.ChoiceField(choices=Joloo.JOLOO_CHOICES, required=True)
    uildverlesen_on = forms.CharField(required=True)
    orj_irsen_on = forms.CharField(required=True)
    hutlugch = forms.ChoiceField(choices=Hutlugch.HUTLUGCH_CHOICES, required=True)
    yavsan_km = forms.CharField(required=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize queryset for xrop field
        if 'motor_bagtaamj' in self.data:
            try:
                motor_bagtaamj_id = int(self.data.get('motor_bagtaamj'))
                self.fields['xrop'].queryset = Xrop.objects.filter(motor_bagtaamj_id=motor_bagtaamj_id)
            except (ValueError, TypeError):
                pass

        # Initialize queryset for orj_irsen_on field
        if 'uildverlesen_on' in self.data:
            try:
                uildverlesen_on_id = int(self.data.get('uildverlesen_on'))
                self.fields['orj_irsen_on'].queryset = OrjIrsenOn.objects.filter(uildverlesen_on_id=uildverlesen_on_id)
            except (ValueError, TypeError):
                pass

        # Initialize queryset for hutlugch field
        if 'orj_irsen_on' in self.data:
            try:
                orj_irsen_on_id = int(self.data.get('orj_irsen_on'))
                self.fields['hutlugch'].queryset = Hutlugch.objects.filter(orj_irsen_on_id=orj_irsen_on_id)
            except (ValueError, TypeError):
                pass

        # Initialize queryset for yavsan_km field
        if 'hutlugch' in self.data:
            try:
                hutlugch_id = int(self.data.get('hutlugch'))
                self.fields['yavsan_km'].queryset = YavsanKm.objects.filter(hutlugch_id=hutlugch_id)
            except (ValueError, TypeError):
                pass
