from django.urls import path
from . import views
app_name="voice"

urlpatterns = [
    path("", views.record, name="index"),
    path("record/detail/<uuid:id>/", views.record_detail, name="record_detail"),
    path("generated/",views.generate_audio, name="generated"),

]

