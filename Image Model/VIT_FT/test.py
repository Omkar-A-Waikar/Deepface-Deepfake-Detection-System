import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Set the path to the directory with the trained model
model_path = "F:/FF_Dataset/outputs/final_model"
threshold=0.7
# Function to load the model and image processor
def load_model(model_path):
    """
    Load the model and image processor.
    :param model_path: Path to the model directory
    :return: Loaded model and image processor
    """
    model = ViTForImageClassification.from_pretrained(model_path)
    processor = ViTImageProcessor.from_pretrained(model_path)
    return model, processor

# Load the model and image processor
model, feature_extractor = load_model(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to test a single image
def test_single_image(image_path):
    """
    Test a single image and display the prediction result with confidence score.
    :param image_path: Path to the test image
    """
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)

    # Model inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits

    # Calculate probabilities
    probs = F.softmax(logits, dim=-1)
    fake_prob = probs[0][0].item()  # Assuming the 0th index is for FAKE
    real_prob = probs[0][1].item()  # Assuming the 1st index is for REAL

    confidence, predicted_class_idx = torch.max(probs, dim=-1)
    confidence_score = confidence.item() * 100  # Convert to percentage
    if fake_prob > threshold:
        predicted_label = 'FAKE'
        confidence_score = fake_prob * 100
    else:
        predicted_label = 'REAL'
        confidence_score = real_prob * 100
    # Determine label
   #predicted_label = 'REAL' if predicted_class_idx.item() == 1 else 'FAKE'

    print(f"Predicted label: {predicted_label} ({confidence_score:.2f}%)")

    # Display the image and predicted label with confidence score
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Prediction: {predicted_label} ({confidence_score:.2f}%)")
    plt.show()

# Set the path to the test image (replace with the actual image path)
test_image_path = "F:/python project/deepfakedetection/real2.jpg"  # Replace with the path of the image to test
test_single_image(test_image_path)
