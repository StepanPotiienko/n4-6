from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("export/csv/", views.export_csv, name="export_csv"),
]
