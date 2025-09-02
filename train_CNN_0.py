"""
CNN Training Code
Version: 0
Dataset: keggle
accuracy: 0.2318 - loss: 2.8027 - val_accuracy: 0.2111 - val_loss: 2.9446
Epoch: 23/50
Batch-size: 64
"""

import os
import pandas as pd
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import pickle
from collections import Counter

# ======================= CONFIG =======================
CSV_PATH = "C:/Users/zizhu/.ssh/PS_Project/.kaggle/archive/train_labels.csv"
TRAIN_FOLDER = "C:/Users/zizhu/.ssh/PS_Project/.kaggle/archive/train/train"
TEST_FOLDER = "C:/Users/zizhu/.ssh/PS_Project/.kaggle/archive/test/test"
TARGET_SIZE = (128, 128)
BATCH_SIZE = 64

# ======================= LOADER =======================
def load_dataset(csv_path, train_folder, test_folder, target_size=(128, 128)):
    df = pd.read_csv(csv_path)

    # Drop classes with < 2 samples
    class_counts = Counter(df["class"])
    drop_classes = {cls for cls, count in class_counts.items() if count < 2}
    df = df[~df["class"].isin(drop_classes)]
    print(f"Dropped {len(drop_classes)} classes with < 2 samples.")

    images = []
    labels = []

    for idx, row in df.iterrows():
        filename = row['filename']
        img_path = os.path.join(train_folder, filename)
        if not os.path.exists(img_path):
            img_path = os.path.join(test_folder, filename)
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        try:
            xmin = int(row['xmin'])
            ymin = int(row['ymin'])
            xmax = int(row['xmax'])
            ymax = int(row['ymax'])
            cropped = img[ymin:ymax, xmin:xmax]
        except:
            continue

        if cropped.size == 0:
            continue

        resized = cv2.resize(cropped, target_size)
        normalized = resized / 255.0
        images.append(normalized)
        labels.append(row['class'])

    return np.array(images), np.array(labels)

# ======================= MAIN =======================
if __name__ == "__main__":
    X, y = load_dataset(CSV_PATH, TRAIN_FOLDER, TEST_FOLDER, target_size=TARGET_SIZE)

    if len(X) == 0 or len(y) == 0:
        print("❌ No data loaded.")
        exit(1)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    num_classes = len(label_encoder.classes_)
    print(f"✅ Loaded {len(X)} samples with {num_classes} classes.")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    # Model
    model = models.Sequential([
        layers.Input(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
              validation_data=(X_test, y_test),
              epochs=50,
              callbacks=[early_stop])

    model.save("model_CNN_0.keras")

