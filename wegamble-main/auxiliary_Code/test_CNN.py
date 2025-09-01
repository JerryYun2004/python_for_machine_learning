import os
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle

def load_test_data(csv_path, image_folder, target_size=(224, 224)):
    df = pd.read_csv(csv_path)
    images = []
    labels = []
    file_names = []

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
        normalized = resized / 255.0

        images.append(normalized)
        labels.append(row['class'])
        file_names.append(row['filename'])

    return np.array(images), np.array(labels), file_names


if __name__ == "__main__":
    # Paths
    test_csv = "C:/Users/zizhu/.ssh/PS_Project/.kaggle/archive/test_labels.csv"
    test_folder = "C:/Users/zizhu/.ssh/PS_Project/.kaggle/archive/test/test"
    model_path = r"C:\Users\zizhu\.ssh\ps_Project\card_Master\model_CNN.keras"  

    # Load test data
    X_test, y_test_text, filenames = load_test_data(test_csv, test_folder)

    if len(X_test) == 0:
        print("❌ No test data loaded. Check your test CSV and image folder.")
        exit(1)

    # Load trained model
    model = tf.keras.models.load_model(model_path)

    # Load the trained label encoder to ensure consistency
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    # Filter out test samples with unseen labels
    valid_indices = [i for i, label in enumerate(y_test_text) if label in label_encoder.classes_]
    X_test = X_test[valid_indices]
    y_test_text = [y_test_text[i] for i in valid_indices]
    filenames = [filenames[i] for i in valid_indices]

    # Encode filtered labels
    y_test = label_encoder.transform(y_test_text)

    # Predict
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)

    # Report
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Print some sample results
    for i in range(min(5, len(X_test))):
        pred_class = label_encoder.inverse_transform([y_pred[i]])[0]
        true_class = y_test_text[i]
        print(f"{filenames[i]} → Predicted: {pred_class}, True: {true_class}")