import os
from torch.utils.data import Dataset
from PIL import Image
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, feature_extractor, label_mapping={'REAL': 1, 'FAKE': 0}):
        """
        Initializes the dataset.
        :param img_dir: Root directory containing images with subdirectories 'REAL' and 'FAKE'.
        :param feature_extractor: Feature extractor for preprocessing images.
        :param label_mapping: Dictionary mapping folder names to labels, e.g., {'REAL': 1, 'FAKE': 0}.
        """
        self.img_dir = img_dir
        self.feature_extractor = feature_extractor
        self.label_mapping = label_mapping
        self.img_files = []
        self.img_labels = []

        # Traverse 'REAL' and 'FAKE' directories
        for label_dir in label_mapping.keys():
            dir_path = os.path.join(img_dir, label_dir)
            files = os.listdir(dir_path)
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg')):  # Only process image files
                    self.img_files.append(os.path.join(dir_path, file))
                    self.img_labels.append(label_mapping[label_dir])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.img_labels[idx]
        # Preprocess image with feature extractor
        features = self.feature_extractor(images=image, return_tensors="pt")
        return {"pixel_values": features['pixel_values'].squeeze(), "labels": torch.tensor(label)}

def load_model(model_id="google/vit-base-patch16-224-in21k", num_labels=2):
    """
    Loads a pretrained ViT model and adjusts the output layer for binary classification.
    :param model_id: Name or path of the ViT model.
    :param num_labels: Number of output labels.
    """
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_id)
    model = ViTForImageClassification.from_pretrained(model_id, num_labels=num_labels)
    return model, feature_extractor

def predict_image(image_path, model, feature_extractor, device):
    """
    Predicts the label of a single image.
    :param image_path: Path to the image file.
    :param model: The ViT model.
    :param feature_extractor: Feature extractor for preprocessing.
    :param device: Device to run the model on (CPU or GPU).
    """
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = 'REAL' if predicted_class_idx == 1 else 'FAKE'
    return predicted_label
