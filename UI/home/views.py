# home/views.py
from django.shortcuts import render

def home(request):
    return render(request, 'home.html')

def photo_deepfake(request):
    return render(request, 'home/photo_deepfake.html')

def audio_deepfake(request):
    return render(request, 'home/audio_deepfake.html')

def faq_photo(request):
    return render(request, 'faq_photo.html')

def faq_audio(request):
    return render(request, 'faq_audio.html')

def contact(request):
    return render(request, 'home/contact.html')




