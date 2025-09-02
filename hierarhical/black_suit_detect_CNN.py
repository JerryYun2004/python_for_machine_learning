import os
import pandas as pd
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# ======================= CONFIG =======================
TARGET_SIZE = (128, 128)
BATCH_SIZE = 64

# ======================= GENERATOR =======================
class CardDatasetGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, label_encoder, batch_size=64, image_size=(128, 128), shuffle=True):
        self.df = dataframe.copy()
        self.label_encoder = label_encoder
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indices]

        images, labels = [], []

        for _, row in batch_df.iterrows():
            filename, label, root = row.get("filename"), row.get("class"), row.get("root")
            if pd.isna(filename) or pd.isna(label) or pd.isna(root):
                continue

            img_path = os.path.join(root, filename)
            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            if len(img.shape) == 2 or img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = cv2.resize(img, self.image_size)
            img = img.astype(np.float32) / 255.0

            images.append(img)
            labels.append(label)

        if not images:
            raise ValueError(f"No valid images found in batch {index}.")

        X = np.array(images)
        y = self.label_encoder.transform(labels)
        return X, y

# ======================= MAIN =======================
def prepare_black_suits_model():
    df1 = pd.read_csv(r"C:\Users\zizhu\.ssh\ps_Project\.kaggle\archive\train_labels.csv")
    df2 = pd.read_csv(r"C:\Users\zizhu\.ssh\ps_Project\tree_cards_database\cards.csv")

    df2 = df2.rename(columns={"filename": "filename", "class": "class"})
    df2 = df2[["filename", "class"]]

    # Keep only spades and clubs
    df1 = df1[df1["class"].str.contains("spades|clubs")]
    df2 = df2[df2["class"].str.contains("spades|clubs")]

    # Replace full class name with just suit name
    df1["class"] = df1["class"].apply(lambda x: "spades" if "spades" in x else "clubs")
    df2["class"] = df2["class"].apply(lambda x: "spades" if "spades" in x else "clubs")

    df1["root"] = df1["filename"].apply(
        lambda x: r"C:\Users\zizhu\.ssh\ps_Project\.kaggle\archive\train\train" if os.path.exists(
            os.path.join(r"C:\Users\zizhu\.ssh\ps_Project\.kaggle\archive\train\train", x)) else
        r"C:\Users\zizhu\.ssh\ps_Project\.kaggle\archive\test\test")

    df2["root"] = r"C:\Users\zizhu\.ssh\ps_Project\tree_cards_database"

    df = pd.concat([df1, df2], ignore_index=True)

    label_encoder = LabelEncoder()
    label_encoder.fit(df["class"])

    with open("label_encoder_black.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_split = int(0.8 * len(df))
    train_df, val_df = df.iloc[:val_split], df.iloc[val_split:]

    train_gen = CardDatasetGenerator(train_df, label_encoder=label_encoder, batch_size=BATCH_SIZE, image_size=TARGET_SIZE)
    val_gen = CardDatasetGenerator(val_df, label_encoder=label_encoder, batch_size=BATCH_SIZE, image_size=TARGET_SIZE, shuffle=False)

    model = models.Sequential([
        layers.Input(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    model.fit(train_gen,
              validation_data=val_gen,
              epochs=50,
              callbacks=[early_stop],
              verbose=1)

    model.save("model_CNN_spades_vs_clubs.keras")

prepare_black_suits_model()
