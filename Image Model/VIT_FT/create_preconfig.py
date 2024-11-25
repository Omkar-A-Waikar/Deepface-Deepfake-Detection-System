from transformers import ViTFeatureExtractor

# 
model_path = "F:/FF_Dataset/outputs/final_model"

# load feature_extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# create preprocessor_config.json
feature_extractor.save_pretrained(model_path)

print("preprocessor_config.json has been generated in the model directory.")
