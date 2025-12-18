import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
model = None
def get_model():
    global model
    if model == None:
        print("Loading MobileNetV2 for CNN feature extraction...")
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        model = Model(inputs=base_model.input, outputs=x)
        print("✓ MobileNetV2 CNN model loaded successfully (1280-d feature vector)")
    return model

def extract_cnn_features(image):
    img_resized = cv2.resize(image,(224 , 224))
    if len(img_resized.shape) == 2 or len(img_resized.shape) == 1:
        if len(img_resized.shape) == 2:
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        elif img_resized.shape[2] == 1:
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_preprocessed = preprocess_input(np.expand_dims(img_rgb.astype(np.float32), axis=0))
    model = get_model()
    features = model.predict(img_preprocessed, verbose=0)
    return features.flatten()

def extract_features(image):
    cnn_feats = extract_cnn_features(image)
    combined_features = np.concatenate([cnn_feats])
    return combined_features

