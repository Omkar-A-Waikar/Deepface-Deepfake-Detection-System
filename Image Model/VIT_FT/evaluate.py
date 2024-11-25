import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from torch.utils.data import DataLoader
from Vision_Transformer import load_model, CustomImageDataset
import os

# Set model and dataset paths
model_path = "F:/FF_Dataset/outputs/final_model"
test_data_dir = "F:/FF_Dataset/processed_frames/test"

# Load model and feature extractor
model, feature_extractor = load_model(model_id=model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the test dataset
test_dataset = CustomImageDataset(img_dir=test_data_dir, feature_extractor=feature_extractor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Function to perform model evaluation
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(pixel_values)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds

# Run evaluation
true_labels, pred_labels = evaluate(model, test_loader, device)

# Calculate performance metrics
accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels, average='binary')
recall = recall_score(true_labels, pred_labels, average='binary')
f1 = f1_score(true_labels, pred_labels, average='binary')

print("Evaluation Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(true_labels, pred_labels, target_names=["FAKE", "REAL"]))



'''
Evaluation Results 1.0:
Accuracy: 0.8776
Precision: 0.9333
Recall: 0.0386
F1 Score: 0.0741

Detailed Classification Report:
              precision    recall  f1-score   support

        FAKE       0.88      1.00      0.93      4992
        REAL       0.93      0.04      0.07       726

    accuracy                           0.88      5718
   macro avg       0.91      0.52      0.50      5718
weighted avg       0.88      0.88      0.83      5718


Evaluation Results 2.0:
Accuracy: 0.7343
Precision: 0.0983
Recall: 0.1336
F1 Score: 0.1133

Detailed Classification Report:
              precision    recall  f1-score   support

        FAKE       0.87      0.82      0.84      4992
        REAL       0.10      0.13      0.11       726

    accuracy                           0.73      5718
   macro avg       0.48      0.48      0.48      5718
weighted avg       0.77      0.73      0.75      5718

'''
