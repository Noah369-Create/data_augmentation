## Monkeypox Image Dataset Balancing and Augmentation 
#### Explanation of Data Augmentation Process:

- ImageDataGenerator:

It is set up to apply random transformations (rotations, shifts, zooms, flips) to the images for augmentation.

- Augment_images function:

A function handles the generation of augmented images using the ImageDataGenerator's flow() method, saving new images to the respective directory.

- Main loop:

The script loops over each class in your dataset.
It calculates how many augmented images are needed to reach 500 for each class.
It generates the required number of augmented images and saves them in the respective folders.