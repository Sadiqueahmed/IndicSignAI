import os
import zipfile

folder_path = 'ISL_IMAGE.keras'
zip_path = 'ISL_FINAL.keras'

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            # Add file to zip archive with relative path
            zipf.write(file_path, os.path.relpath(file_path, folder_path))

print(f"Successfully zipped {folder_path} into {zip_path}")

import sys
try:
    import tensorflow as tf
    model = tf.keras.models.load_model(zip_path)
    print("Model loaded successfully via zip!")
    print(model.summary())
except Exception as e:
    print(f"Failed to load: {e}")
