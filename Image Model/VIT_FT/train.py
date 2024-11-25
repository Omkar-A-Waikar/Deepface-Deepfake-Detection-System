import os
import torch
from transformers import Trainer, TrainingArguments, default_data_collator
from Vision_Transformer import CustomImageDataset, load_model

# Set the dataset path and model ID
img_dir = "F:/FF_Dataset/processed_frames"
model_id = "google/vit-base-patch16-224-in21k"

# Load the model and feature extractor
model, feature_extractor = load_model(model_id=model_id)

# Set device (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load training and testing datasets
train_dataset = CustomImageDataset(img_dir=img_dir + "/train", feature_extractor=feature_extractor)
test_dataset = CustomImageDataset(img_dir=img_dir + "/test", feature_extractor=feature_extractor)


# Calculate class weights based on the number of files in each class folder
def calculate_class_weights_from_folders(train_dir):
    real_dir = os.path.join(train_dir, "REAL")
    fake_dir = os.path.join(train_dir, "FAKE")

    # Count the number of images in each folder
    num_real = len(os.listdir(real_dir))
    num_fake = len(os.listdir(fake_dir))
    total_count = num_real + num_fake

    # Compute weights inversely proportional to class frequencies
    weight_real = total_count / (2 * num_real)
    weight_fake = total_count / (2 * num_fake)
    return torch.tensor([weight_fake, weight_real]).to(device)


# Compute class weights
class_weight = calculate_class_weights_from_folders(img_dir + "/train")


# Define Custom Trainer to use weighted loss
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # Apply weighted cross entropy loss
        loss = torch.nn.functional.cross_entropy(logits, labels, weight=class_weight)
        return (loss, outputs) if return_outputs else loss



# Define training parameters with optimized hyperparameters
training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=32,  # Adjusted batch size
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    num_train_epochs=3,  # Increased epochs for better learning
    save_strategy="epoch",
    logging_steps=50,
    learning_rate=3e-5,  # Lower learning rate for better convergence
    save_total_limit=3,  # Keep only the 3 most recent checkpoints
    remove_unused_columns=False,
    load_best_model_at_end=True,
    fp16=True,  # Enable mixed precision training for faster computation
    weight_decay=0.01  # Apply weight decay to reduce overfitting
)

# Initialize CustomTrainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=default_data_collator,
)

print("Start training")
# Start training
trainer.train()

# Save the final model
trainer.save_model("./outputs/final_model")
print("Training complete. Model saved to ./outputs/final_model.")
