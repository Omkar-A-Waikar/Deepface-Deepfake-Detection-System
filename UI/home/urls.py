from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    # path('faq/photo/', views.faq_photo, name='faq_photo'),
    # path('faq/audio/', views.faq_audio, name='faq_audio'),
    path('contact/', views.contact, name='contact'),
    path('photo-deepfake/', views.photo_deepfake, name='photo_deepfake'),
    path('audio-deepfake/', views.audio_deepfake, name='audio_deepfake'),

]