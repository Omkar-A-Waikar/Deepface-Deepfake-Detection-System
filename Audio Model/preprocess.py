import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import gc
import psutil

def save_spectrogram_image(y, sr, output_path):
    """Generate and save a mel spectrogram image."""
    plt.figure(figsize=(10, 4))
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), sr=sr)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return spectrogram

def log_memory_usage():
    """Log the current memory usage of the process."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / (1024 ** 2):.2f} MB")

def convert_audio_to_images(data_dir, output_dir, batch_size=10):
    """Convert audio files to mel spectrogram images."""
    os.makedirs(output_dir, exist_ok=True)
    files_processed = 0
    for category in ['real', 'fake']:
        category_dir = os.path.join(data_dir, category)
        output_category_dir = os.path.join(output_dir, category)
        os.makedirs(output_category_dir, exist_ok=True)

        file_list = [f for f in os.listdir(category_dir) if f.endswith('.wav')]
        num_files = len(file_list)

        for start in range(0, num_files, batch_size):
            end = min(start + batch_size, num_files)
            batch_files = file_list[start:end]

            for file_name in batch_files:
                try:
                    file_path = os.path.join(category_dir, file_name)
                    y, sr = librosa.load(file_path)
                    output_path = os.path.join(output_category_dir, file_name.replace('.wav', '.png'))
                    save_spectrogram_image(y, sr, output_path)
                    files_processed += 1
                except Exception as e:
                    print(f"Error processing file {file_name}: {e}")

                log_memory_usage()
                gc.collect()  # Clear unused memory

    print(f"Total files processed: {files_processed}")

def load_images(data_dir):
    """Load images from the specified directory and assign labels."""
    data = []
    labels = []
    for category in ['real', 'fake']:
        category_dir = os.path.join(data_dir, category)
        for file_name in os.listdir(category_dir):
            if file_name.endswith('.png'):
                file_path = os.path.join(category_dir, file_name)
                img = load_img(file_path, target_size=(224, 224))
                img_array = img_to_array(img)
                data.append(img_array)
                labels.append(0 if category == 'real' else 1)
    return np.array(data), np.array(labels)

def preprocess_data(training_data, testing_data, validation_data):
    """Preprocess images for model input."""
    from keras.applications.vgg16 import preprocess_input
    return preprocess_input(training_data), preprocess_input(testing_data), preprocess_input(validation_data)

# Define paths
base_path = r'/workspace/mnt/data/for-norm'
training_path = os.path.join(base_path, 'training')
testing_path = os.path.join(base_path, 'testing')
validation_path = os.path.join(base_path, 'validation')
training_images_path = os.path.join(base_path, 'training_images')
testing_images_path = os.path.join(base_path, 'testing_images')
validation_images_path = os.path.join(base_path, 'validation_images')

# Convert audio files to images
convert_audio_to_images(training_path, training_images_path, batch_size=10)
convert_audio_to_images(testing_path, testing_images_path, batch_size=10)
convert_audio_to_images(validation_path, validation_images_path, batch_size=10)

# Load and preprocess images
training_data, training_labels = load_images(training_images_path)
testing_data, testing_labels = load_images(testing_images_path)
validation_data, validation_labels = load_images(validation_images_path)

# Preprocess the data
training_data, testing_data, validation_data = preprocess_data(training_data, testing_data, validation_data)