import os
import cv2
import numpy as np
import joblib
import pandas as pd

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D


CLASSES = {
    0: "glass",
    1: "paper",
    2: "cardboard",
    3: "plastic",
    4: "metal",
    5: "trash",
    6: "unknown",
}
cnn_model = None


def get_model():
    global cnn_model
    if cnn_model is None:
        print("Loading MobileNetV2 for CNN feature extraction...")
        base_model = MobileNetV2(
            weights="imagenet", include_top=False, input_shape=(224, 224, 3)
        )
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        cnn_model = Model(inputs=base_model.input, outputs=x)
        print("✓ MobileNetV2 CNN model loaded successfully (1280-d feature vector)")
    return cnn_model


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


def predict(dataFilePath, bestModelPath):
    model = joblib.load(bestModelPath)

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
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


def predict_and_save_to_excel(folder_path, model_path, output_excel_path):
    model = joblib.load(model_path)

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    image_files = [
        f
        for f in sorted(os.listdir(folder_path))
        if f.lower().endswith(valid_extensions)
    ]

    results = []

    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            predicted_label = "unknown"
        else:
            p_img = preprocess(img)
            feat = extract_features(p_img).reshape(1, -1)

            probs = model.predict_proba(feat)
            max_prob = np.max(probs)
            pred_idx = np.argmax(probs)
            if max_prob < 0.5:
                predicted_label = "unknown"
            else:
                class_id = model.classes_[pred_idx]
                predicted_label = CLASSES.get(int(class_id), "unknown")

        image_name = os.path.splitext(img_file)[0]
        results.append({"ImageName": image_name, "predictedlabel": predicted_label})

    df = pd.DataFrame(results)
    df.to_excel(output_excel_path, index=False)
    print(f"Predictions saved to {output_excel_path}")

if __name__ == "__main__":
    data_path = "ML_Project/testdata"
    model_path = "ML_Project/models/svm_model.pkl"

    preds = predict(data_path, model_path)
    print(preds)

    predict_and_save_to_excel("ML_Project/sample", model_path, "output2.xlsx")
