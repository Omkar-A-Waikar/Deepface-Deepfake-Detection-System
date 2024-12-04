from django.shortcuts import render
from django.http import JsonResponse
from .forms import ImageUploadForm
from .model_utils import test_single_image  # 假设你将 test_single_image 函数放在 utils.py 中

def classify_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            # 调用你的模型预测函数
            predicted_label, confidence_score = test_single_image(image)
            return JsonResponse({
                'predicted_label': predicted_label,
                'confidence_score': confidence_score
            })
    else:
        form = ImageUploadForm()
    return render(request, 'image_classification/upload.html', {'form': form})