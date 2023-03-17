from django.db import models
from django.urls.base import reverse
import uuid

class Record(models.Model):
    id=models.UUIDField(primary_key=True,default=uuid.uuid4,editable=False)
    voice_record = models.FileField(upload_to="records")
    name=models.CharField(max_length=100)
    class Meta:
        verbose_name="Record"
        verbose_name_plural="Records"
    def __str__(self):
        return str(self.id)

    def get_absolute_url(self):
        return reverse("core:record_detail", kwargs={"id": str(self.id)})
class Generated(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    generated_voice=models.FileField(upload_to="generated")
    name=models.CharField(max_length=200)

    class Meta:
        verbose_name = "Generated"
        verbose_name_plural = "Generateds"

    def __str__(self):
        return str(self.id)

    def get_absolute_url(self):
        return reverse("core:generated", kwargs={"id": str(self.id)})
