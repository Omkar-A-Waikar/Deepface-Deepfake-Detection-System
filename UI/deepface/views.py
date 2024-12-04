from django.shortcuts import render
from django.http import JsonResponse
from .forms import ImageUploadForm
from .model_utils import ImageClassifier  # Import the ImageClassifier class

# Initialize the classifier once to avoid reloading the model for each request
model_path = "F:/FF_Dataset/outputs/final_model"
classifier = ImageClassifier(model_path)

def classify_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            # Use the classifier's predict method
            predicted_label, confidence_score = classifier.predict(image)
            return JsonResponse({
                'predicted_label': predicted_label,
                'confidence_score': confidence_score
            })
    else:
        form = ImageUploadForm()
    return render(request, 'image_classification/upload.html', {'form': form})
