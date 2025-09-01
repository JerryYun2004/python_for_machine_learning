import os
import pandas as pd
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def load_dataset_from_csv(csv_path, image_folder, target_size=(64, 64)):
    """
    Load image crops and labels from CSV annotation file.

    Args:
        csv_path (str): Path to CSV file with annotations.
        image_folder (str): Folder containing image files.
        target_size (tuple): Output size for resized crops.

    Returns:
        X: NumPy array of flattened image crops
        y: List of class labels
    """
    df = pd.read_csv(csv_path)
    images = []
    labels = []

    for idx, row in df.iterrows():
        img_path = os.path.join(image_folder, row['filename'])
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        xmin = int(row['xmin'])
        ymin = int(row['ymin'])
        xmax = int(row['xmax'])
        ymax = int(row['ymax'])

        cropped = img[ymin:ymax, xmin:xmax]
        if cropped.size == 0:
            continue

        resized = cv2.resize(cropped, target_size)
        flattened = resized.flatten() / 255.0  # Normalize to [0, 1]

        images.append(flattened)
        labels.append(row['class'])

    return np.array(images), np.array(labels)


if __name__ == "__main__":
    # Paths
    csv_path = "C:/Users/zizhu/.ssh/PS_Project/.kaggle/archive/train_labels.csv"
    image_folder = r"C:\Users\zizhu\.ssh\ps_Project\.kaggle\archive\train\train"  # fixed

    X, y = load_dataset_from_csv(csv_path, image_folder)

    if len(X) == 0 or len(y) == 0:
        raise ValueError("❌ No data loaded. Check the image folder path and CSV.")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42)
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    idx = 0
    predicted_class = label_encoder.inverse_transform([y_pred[idx]])[0]
    true_class = label_encoder.inverse_transform([y_test[idx]])[0]
    print(f"Sample Prediction: {predicted_class} (True: {true_class})")
