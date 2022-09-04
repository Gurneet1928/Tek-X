from django.db import models

# Create your models here.


class Video(models.Model):
    caption = models.CharField(max_length=100)
    video = models.FileField(upload_to="video/%y")
    summary = models.CharField(max_length=5000, null=True, blank=False)

    def __str__(self):
        return self.caption
