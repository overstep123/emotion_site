from django.db import models

class Img(models.Model):
    img_url = models.ImageField(upload_to='img')

class Video(models.Model):
    video_url = models.FileField(upload_to="vdo")
