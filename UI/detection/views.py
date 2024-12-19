import os
import torch
import torch.nn as nn
from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from torchvision import transforms
from huggingface_hub import hf_hub_download
from PIL import Image
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from .forms import SignUpForm, LoginForm

from .forms import ImageUploadForm

# Home View
def home(request):
    return render(request, 'home.html')

def contact(request):
    return render(request, 'contact.html')

# Sign Up View
def signup_view(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()  # Save the user instance
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')  # Password1 for UserCreationForm
            user = authenticate(request, username=username, password=password)  # Authenticate the new user
            if user:
                login(request, user)  # Log the user in
                messages.success(request, f"Welcome, {username}! Your account has been created.")
                return redirect('home')  # Redirect to home page after login
    else:
        form = SignUpForm()
    return render(request, 'signup.html', {'form': form})


# Login View
def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(request, username=username, password=password)
            if user:
                login(request, user)
                messages.success(request, f"Welcome back, {username}!")
                return redirect('home')
            else:
                messages.error(request, "Invalid username or password.")
    else:
        form = LoginForm()
    return render(request, 'login.html', {'form': form})

# Logout View
def logout_view(request):
    logout(request)
    messages.info(request, "You have been logged out.")
    return redirect('login')

# Use non-GUI backend for Matplotlib
matplotlib.use("Agg")


# Image 
# Load the trained model for image detection
IMAGE_MODEL_PATH = os.path.join(settings.BASE_DIR, "model/model.h5")
image_model = load_model(IMAGE_MODEL_PATH)

# Helper function to preprocess an uploaded image
def preprocess_image(uploaded_image):
    img = image.load_img(uploaded_image, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# View for uploading an image and making predictions
def image_upload(request):
    result = None
    confidence = None
    uploaded_image_url = None
    file_name = None  # Initialize file name variable

    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = request.FILES["image"]
            file_name = uploaded_image.name  # Get the file name

            # Save the uploaded image
            save_path = os.path.join(settings.MEDIA_ROOT, "images", uploaded_image.name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb+") as destination:
                for chunk in uploaded_image.chunks():
                    destination.write(chunk)
            uploaded_image_url = f"{settings.MEDIA_URL}images/{uploaded_image.name}"

            # Preprocess and predict
            processed_image = preprocess_image(save_path)
            prediction = image_model.predict(processed_image)[0][0]
            confidence = round(float(prediction) * 100, 2)

            # Determine result
            if prediction < 0.5:
                result = "Real Image"
                confidence = 100 - confidence  # Adjust confidence for 'Real'
            else:
                result = "Deepfake Detected"

            return render(request, "upload_image.html", {
                "form": form,
                "result": result,
                "confidence": confidence,
                "uploaded_image_url": uploaded_image_url,
                "file_name": file_name  # Pass file name to the template
            })
    else:
        form = ImageUploadForm()

    return render(request, "upload_image.html", {"form": form})


# audio
class CustomVGG16(nn.Module):
    def __init__(self):
        super(CustomVGG16, self).__init__()
        from torchvision.models import vgg16
        self.base_model = vgg16(weights='DEFAULT')
        for param in self.base_model.features.parameters():
            param.requires_grad = False
        self.base_model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.base_model(x)

# Load the pretrained audio model
repo_id = "ArmaanDhande/Deepface"
filename = "best_model.pth"
local_model_path = hf_hub_download(repo_id=repo_id, filename=filename)
audio_model = CustomVGG16()
audio_model.load_state_dict(torch.load(local_model_path, map_location=torch.device('cpu')))
audio_model.eval()

# Audio file transformation pipeline
audio_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Helper function to process audio file
def infer_audio_file(file_path):
    # Load audio and generate spectrogram
    y, sr = librosa.load(file_path)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    # Generate spectrogram as an image
    fig = plt.figure(figsize=(2, 2), dpi=112)
    librosa.display.specshow(spectrogram_db, sr=sr)
    plt.axis('off')
    plt.tight_layout()
    canvas = fig.canvas
    canvas.draw()

    # Convert spectrogram to image tensor
    buffer, (width, height) = canvas.print_to_buffer()
    image_np = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 4))
    plt.close(fig)
    image_pil = Image.fromarray(image_np[:, :, :3], 'RGB')
    image_tensor = audio_transform(image_pil).unsqueeze(0)

    # Predict using the audio model
    with torch.no_grad():
        output = audio_model(image_tensor).item()

    # Interpret results
    label = "Fake" if output > 0.2 else "Real"
    confidence = round(output * 100, 2) if output > 0.2 else round((1 - output) * 100, 2)
    return label, confidence

# View for uploading an audio file and making predictions
def audio_detection(request):
    result = None
    confidence = None

    if request.method == "POST" and request.FILES.get("audio_file"):
        # Save uploaded audio file
        audio_file = request.FILES["audio_file"]
        fs = FileSystemStorage()
        saved_file = fs.save(audio_file.name, audio_file)
        file_path = fs.path(saved_file)

        # Run inference
        result, confidence = infer_audio_file(file_path)

    return render(request, "audio_upload.html", {
        "result": result,
        "confidence": confidence,
    })

