import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np

# Directories of each class
data_dir = 'monkeypox_image_dataset'
categories = ['Chickenpox', 'Measles', 'Monkeypox', 'Normal']
target_count = 500  # Target number of images for each class

# Image augmentation setup
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Function to augment and save images
def augment_images(img, save_dir, current_img_count, augment_count):
    x = img_to_array(img)  # Convert image to array
    x = np.expand_dims(x, axis=0)
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=save_dir, save_prefix='aug', save_format='jpg'):
        i += 1
        if i >= augment_count:
            break

# Go through each category and augment images
for category in categories:
    class_dir = os.path.join(data_dir, category)
    images = [f for f in os.listdir(class_dir) if f.endswith(('jpg', 'jpeg', 'png'))]
    current_count = len(images)

    if current_count < target_count:
        augment_count = target_count - current_count
        print(f"Augmenting {augment_count} images for {category}...")

        for img_name in images:
            img_path = os.path.join(class_dir, img_name)
            img = load_img(img_path)  # Load image
            
            # Calculate how many augmentations per image to achieve balance
            aug_per_img = augment_count // current_count
            augment_images(img, class_dir, current_count, aug_per_img)

        # If still need more images (due to rounding)
        current_count = len([f for f in os.listdir(class_dir) if f.endswith(('jpg', 'jpeg', 'png'))])
        while current_count < target_count:
            img_name = images[current_count % len(images)]  # Pick an image cyclically
            img_path = os.path.join(class_dir, img_name)
            img = load_img(img_path)
            augment_images(img, class_dir, current_count, 1)
            current_count += 1

print("Dataset balancing complete.")
