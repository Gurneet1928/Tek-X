from django import forms
from videou.models import Video


class Video_form(forms.ModelForm):
    class Meta:
        model = Video
        exclude = ()
