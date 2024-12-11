# django_project/urls.py
from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("home.urls")), 
    path('accounts/', include('accounts.urls')),
    path('photo_deepfake/', views.photo_deepfake, name='photo_deepfake')
    
]
