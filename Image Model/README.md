DeepFake Detection with Vision Transformer (ViT)


The Fine Tuning Model is in here: 

https://drive.google.com/drive/folders/15BahTRQzD7EUeOBCj2xPFSDa4i3H249w?dmr=1&ec=wgc-drive-hero-goto


This project uses a Vision Transformer (ViT) model (https://github.com/google-research/vision_transformer) to detect DeepFake images by fine-tuning it on a custom dataset created from the FaceForensics DeepFake Detection dataset (https://github.com/ondyari/FaceForensics). Using OpenCV (https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html#a57c0e81e83e60f36c83027dc2a188e80), frames are extracted from real and fake videos to create the real and fake images dataset, which is then used to train a ViT model for AI-based image classification.


Project Overview：

DeepFake detection is essential in various fields to ensure the authenticity of visual content. This project leverages the Vision Transformer (ViT) model from Hugging Face’s Transformers library, fine-tuning it on a custom labeled dataset of real and fake frames extracted from videos in the FaceForensics DeepFake Detection dataset.

The project includes three main components:

1.Data Preprocessing: Extracting frames from real and fake videos to create a images dataset for training. 

2.Model Training: Fine-tuning a ViT model on the custom image dataset. 

To improve model performance on an imbalanced dataset, this project employs a weighted loss function to help the model focus more on the minority class, which is the "REAL" class in this case. In the dataset, the "FAKE" class has significantly more samples than the "REAL" class, causing the model to potentially favor predicting "FAKE" during training. To address this issue, we apply a weighted loss strategy.
Specifically, we calculate weights for each class based on the inverse of their sample frequencies. This means the less frequent "REAL" class is assigned a higher weight, resulting in a larger penalty for misclassifying samples from this class. This approach adjusts the loss function so the model places greater emphasis on learning from the "REAL" class, helping to balance the impact of each class in the training process and ultimately enhancing model performance on imbalanced data.

3.Single Image Testing: Testing the trained model on individual images for classification.

Features：

1. Frame extraction from FaceForensics videos using OpenCV
2. Model fine-tuning using Vision Transformer
3. Single-image testing for detecting DeepFake images
4. GPU support for faster training and inference

Requirements：

Python 3.8 or higher

PyTorch

Hugging Face Transformers

OpenCV

Matplotlib

Torchvision
