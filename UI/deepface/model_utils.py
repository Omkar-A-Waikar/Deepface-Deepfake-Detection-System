import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt

class ImageClassifier:
    def __init__(self, model_path, threshold=0.7):
        self.model = ViTForImageClassification.from_pretrained(model_path)
        self.processor = ViTImageProcessor.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.threshold = threshold

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        fake_prob = probs[0][0].item()
        real_prob = probs[0][1].item()

        if fake_prob > self.threshold:
            predicted_label = 'FAKE'
            confidence_score = fake_prob * 100
        else:
            predicted_label = 'REAL'
            confidence_score = real_prob * 100

        return predicted_label, confidence_score

    def display_prediction(self, image_path):
        """
        Display the image and its prediction result.
        :param image_path: Path to the image file
        """
        # Get prediction results
        predicted_label, confidence_score = self.predict(image_path)
        # Open the image
        image = Image.open(image_path)
        # Display the image with the prediction
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Prediction: {predicted_label} ({confidence_score:.2f}%)")
        plt.show()

# Example usage
if __name__ == '__main__':
    model_path = "F:/FF_Dataset/outputs/final_model"
    test_image_path = "F:/python project/deepfakedetection/real2.jpg"  # Replace with the path of the image to test

    # Initialize the classifier and display prediction
    classifier = ImageClassifier(model_path)
    classifier.display_prediction(test_image_path)
