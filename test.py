import os
import cv2
import numpy as np
import joblib

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D


# ===============================
# CNN Feature Extraction (Singleton)
# ===============================
cnn_model = None


def get_model():
    global cnn_model
    if cnn_model is None:
        print("Loading MobileNetV2 for CNN feature extraction...")
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        cnn_model = Model(inputs=base_model.input, outputs=x)
        print("✓ MobileNetV2 CNN model loaded successfully (1280-d feature vector)")
    return cnn_model


# ===============================
# Image Preprocessing
# ===============================
def preprocess(img):
    # Resize
    img = cv2.resize(img, (224, 224))

    # Noise reduction
    if len(img.shape) == 3:
        img = cv2.fastNlMeansDenoisingColored(
            img, None, h=10, templateWindowSize=7, searchWindowSize=21
        )
    else:
        img = cv2.fastNlMeansDenoising(
            img, None, h=10, templateWindowSize=7, searchWindowSize=21
        )

    # CLAHE
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

    return img


# ===============================
# Feature Extraction
# ===============================
def extract_features(img):
    img_resized = cv2.resize(img, (224, 224))

    if len(img_resized.shape) == 2:
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    img_preprocessed = preprocess_input(
        np.expand_dims(img_rgb.astype(np.float32), axis=0)
    )

    model = get_model()
    features = model.predict(img_preprocessed, verbose=0)
    return features.flatten()


# ===============================
# Prediction Function
# ===============================
def predict(dataFilePath, bestModelPath):
    model = joblib.load(bestModelPath)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_list = [
        os.path.join(dataFilePath, f)
        for f in sorted(os.listdir(dataFilePath))
        if f.lower().endswith(valid_extensions)
    ]

    predictions = []
    confidence_threshold = 0.5

    for img_path in image_list:
        img = cv2.imread(img_path)
        if img is None:
            predictions.append(6)
            continue

        p_img = preprocess(img)
        feat = extract_features(p_img).reshape(1, -1)

        probs = model.predict_proba(feat)
        max_prob = np.max(probs)
        pred_idx = np.argmax(probs)

        if max_prob < confidence_threshold:
            predictions.append(6)
        else:
            class_id = model.classes_[pred_idx]
            predictions.append(int(class_id))

    return predictions


# ===============================
# Main
# ===============================
if __name__ == "__main__":
    data_path = "testdata"
    model_path = "models/svm_model.pkl"

    preds = predict(data_path, model_path)
    print(preds)
