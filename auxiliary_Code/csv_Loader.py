import os
import pandas as pd
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_dataset_from_csv(csv_path, image_folder, target_size=(64, 64)):
    df = pd.read_csv(csv_path)
    images = []
    labels = []

    for idx, row in df.iterrows():
        img_path = os.path.join(image_folder, row['filename'])
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        cropped = img[ymin:ymax, xmin:xmax]

        resized = cv2.resize(cropped, target_size)
        flattened = resized.flatten() / 255.0  # Normalize

        images.append(flattened)
        labels.append(row['class'])

    return np.array(images), np.array(labels)