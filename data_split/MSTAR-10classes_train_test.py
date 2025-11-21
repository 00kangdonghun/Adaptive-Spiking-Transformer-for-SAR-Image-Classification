import os
import shutil
import random

def split_dataset(dataset_path, output_path, train_ratio=0.8, seed=42):
    random.seed(seed)

    classes = os.listdir(dataset_path)
    for cls in classes:
        class_path = os.path.join(dataset_path, cls)
        if not os.path.isdir(class_path):
            continue

        images = [f for f in os.listdir(class_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        random.shuffle(images)

        split_idx = int(len(images) * train_ratio)
        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]

        # Create train folder
        train_dir = os.path.join(output_path, "train", cls)
        os.makedirs(train_dir, exist_ok=True)

        # Create test folder
        test_dir = os.path.join(output_path, "test", cls)
        os.makedirs(test_dir, exist_ok=True)

        # Copy files
        for img in train_imgs:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, img))
        for img in test_imgs:
            shutil.copy(os.path.join(class_path, img), os.path.join(test_dir, img))

        print(f"{cls}: {len(train_imgs)} train, {len(test_imgs)} test")

# Example usage
dataset_path = "/home1/danny472/SAR_project/MSTAR-8classes"  # Original dataset path
output_path = "/home1/danny472/SAR_project/MSTAR_8class_split"  # Output path for the split dataset
split_dataset(dataset_path, output_path, train_ratio=0.8)
