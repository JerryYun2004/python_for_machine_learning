import cv2
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model

TARGET_SIZE = (128, 128)

def load_model_and_encoder(model_path, encoder_path):
    model = load_model(model_path)
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

# Load models and encoders
model_color, encoder_color = load_model_and_encoder(r"C:\Users\zizhu\.ssh\ps_Project\card_Master\thoughtful_approach\model_CNN_red_black.keras", r"C:\Users\zizhu\.ssh\ps_Project\card_Master\thoughtful_approach\label_encoder_red_black.pkl")
model_red_suit, encoder_red_suit = load_model_and_encoder(r"C:\Users\zizhu\.ssh\ps_Project\card_Master\thoughtful_approach\model_CNN_diamonds_vs_clubs.keras", r"C:\Users\zizhu\.ssh\ps_Project\card_Master\thoughtful_approach\label_encoder_red.pkl")
model_black_suit, encoder_black_suit = load_model_and_encoder(r"C:\Users\zizhu\.ssh\ps_Project\card_Master\thoughtful_approach\model_CNN_spades_vs_clubs.keras", r"C:\Users\zizhu\.ssh\ps_Project\card_Master\thoughtful_approach\label_encoder_black.pkl")
model_rank, encoder_rank = load_model_and_encoder(r"C:\Users\zizhu\.ssh\ps_Project\card_Master\thoughtful_approach\model_CNN_rank.keras", r"C:\Users\zizhu\.ssh\ps_Project\card_Master\thoughtful_approach\label_encoder_rank.pkl")

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image from {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, TARGET_SIZE)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)  # shape: (1, 128, 128, 3)

def predict_label(model, encoder, image_array):
    prediction = model.predict(image_array, verbose=0)
    pred_index = np.argmax(prediction, axis=1)[0]
    return encoder.inverse_transform([pred_index])[0]

def predict_card_attributes(img_path):
    image_array = preprocess_image(img_path)

    # Step 1: Color Detection
    color = predict_label(model_color, encoder_color, image_array)

    # Step 2: Suit Detection based on color
    if color == "red":
        suit = predict_label(model_red_suit, encoder_red_suit, image_array)
    elif color == "black":
        suit = predict_label(model_black_suit, encoder_black_suit, image_array)
    else:
        raise ValueError(f"Unexpected color class: {color}")

    # Step 3: Rank Detection
    rank = predict_label(model_rank, encoder_rank, image_array)

    return {
        "color": color,
        "suit": suit,
        "rank": rank
    }


if __name__ == "__main__":
    result = predict_card_attributes(r"C:\Users\zizhu\.ssh\ps_Project\Copag11.jpg")
    print(f"Color: {result['color']}")
    print(f"Suit: {result['suit']}")
    print(f"Rank: {result['rank']}")