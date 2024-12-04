from django.shortcuts import render
from django.http import JsonResponse
from .forms import ImageUploadForm
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
from django.core.files.storage import FileSystemStorage
from .model_utils import ImageClassifier
import logging
import time
logger = logging.getLogger(__name__)
# Load the model and processor
model_path = 'F:/python project/Team4-SE-main/final_model'  # Use TorchScript model
model = torch.jit.load(model_path)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

def photo_deepfake(request):
    if request.method == 'POST' and request.FILES['image_file']:
        logger.info("Received POST request with file.")
        image_file = request.FILES['image_file']
        fs = FileSystemStorage()
        filename = fs.save(image_file.name, image_file)
        uploaded_file_url = fs.url(filename)

        start_time = time.time()

        # use local model to predict
        model_path = 'F:/python project/Team4-SE-main/final_model'
        classifier = ImageClassifier(model_path)
        try:
            predicted_label, confidence_score = classifier.predict(fs.path(filename))
            if not predicted_label:  # if prediction is empty
                predicted_label = 'FAKE'
                confidence_score = 0.0
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            predicted_label = 'FAKE'
            confidence_score = 0.0

        # time on prediction
        end_time = time.time()
        logger.info(f"Prediction took {end_time - start_time:.2f} seconds.")

        result = f"{predicted_label} ({confidence_score:.2f}%)"

        return render(request, 'home/photo_deepfake.html', {
            'uploaded_file_url': uploaded_file_url,
            'result': result
        })
    return render(request, 'home/photo_deepfake.html')
def classify_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            image = Image.open(image).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()

            return JsonResponse({
                'predicted_class': predicted_class,
                'probabilities': probabilities.tolist()
            })
    else:
        form = ImageUploadForm()
    return render(request, 'image_classification/upload.html', {'form': form})

def predict_deepfake(image_path):
    model_path = 'F:/python project/Team4-SE-main/final_model'
    classifier = ImageClassifier(model_path)
    predicted_label, confidence_score = classifier.predict(image_path)
    return f"{predicted_label} ({confidence_score:.2f}%)"
