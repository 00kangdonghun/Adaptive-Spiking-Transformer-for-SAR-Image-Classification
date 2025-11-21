import os
import shutil
import random

# Original dataset path
dataset_dir = "/home1/danny472/SAR_project/EuroSAT-RGB"
# Output path for the split dataset
output_dir = "/home1/danny472/SAR_project/EuroSAT"

# train, test ratio
train_ratio = 0.8
test_ratio = 0.2

# Create train/test folders
for split in ["train", "test"]:
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)

# Iterate over class folders
for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    # Create train/test class folders
    for split in ["train", "test"]:
        split_class_dir = os.path.join(output_dir, split, class_name)
        os.makedirs(split_class_dir, exist_ok=True)

    # Load image file list
    images = os.listdir(class_path)
    random.shuffle(images)

    # train/test split
    split_index = int(len(images) * train_ratio)
    train_files = images[:split_index]
    test_files = images[split_index:]

    # Copy files
    for file in train_files:
        shutil.copy(os.path.join(class_path, file),
                    os.path.join(output_dir, "train", class_name, file))
    for file in test_files:
        shutil.copy(os.path.join(class_path, file),
                    os.path.join(output_dir, "test", class_name, file))

print("Dataset splitting complete!")
