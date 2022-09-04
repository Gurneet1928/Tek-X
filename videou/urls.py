from django.urls import path
from videou import views

urlpatterns = [
    path('', views.index, name="index"),
    path('main/upload/', views.upload, name="upload"),
]
