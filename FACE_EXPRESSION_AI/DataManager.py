import os, shutil, numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, save_img

DATASET_PATH = r'train'                # your current folder
OUTPUT_PATH  = r'train_balanced'       # new balanced datasets copy
TARGET_SIZE  = (48, 48)                # change if needed for now its 48x48 same as the model input 
COLOR_MODE   = "grayscale"             # or "rgb"
AUG_PER_IMG  = 2    

#####################
# Script to balance the dataset by augmenting images
# It will create a new folder with balanced classes
# For non biased datasets, it will augment the smaller classes
# For biased datasets, it will augment the smaller classes until they match the largest class
###################### 

os.makedirs(OUTPUT_PATH, exist_ok=True)
classes = [c for c in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, c))]
class_counts = {c: len(os.listdir(os.path.join(DATASET_PATH, c))) for c in classes}
max_count = max(class_counts.values())
print("Class counts:", class_counts, "\nTarget per class:", max_count)

datagen = ImageDataGenerator(rescale=1./255,rotation_range=30,shear_range=0.3,zoom_range=0.3,width_shift_range=0.4,height_shift_range=0.4,horizontal_flip=True,vertical_flip=True)

for c in classes:
    in_dir  = os.path.join(DATASET_PATH, c)
    out_dir = os.path.join(OUTPUT_PATH, c)
    os.makedirs(out_dir, exist_ok=True)

    # copy originals
    files = [f for f in os.listdir(in_dir)]
    for f in files:
        shutil.copy(os.path.join(in_dir, f), os.path.join(out_dir, f))

    # augment until we reach max_count
    cur = class_counts[c]
    idx = 0
    while cur < max_count:
        img_path = os.path.join(in_dir, files[idx % len(files)])
        img = load_img(img_path, target_size=TARGET_SIZE, color_mode=COLOR_MODE)
        x = img_to_array(img)[None, ...]
        for _ in range(AUG_PER_IMG):
            if cur >= max_count: break
            aug = next(datagen.flow(x, batch_size=1))[0]
            save_img(os.path.join(out_dir, f"aug_{cur}.png"), aug)
            cur += 1
        idx += 1

print("Balanced dataset saved to:", OUTPUT_PATH)