import cv2
import numpy as np
import joblib

from src import feature_extraction

CLASSES = {
    0: "Glass",
    1: "Paper",
    2: "Cardboard",
    3: "Plastic",
    4: "Metal",
    5: "Trash",
    6: "Unknown"
}

def load_inference_model(model_path):
    model = joblib.load(model_path)
    print(f"Loaded classifier model successfully.")
    print(f"Model type: {type(model).__name__}")
    return model

def predict_frame(model, frame, threshold=0.3):
    h, w = frame.shape[:2]
    box_size = 300
    x1 = (w - box_size) // 2
    y1 = (h - box_size) // 2
    x2 = x1 + box_size
    y2 = y1 + box_size
    roi = frame[y1:y2, x1:x2]
    feat = feature_extraction.extract_features(roi)
    feat = feat.reshape(1, -1)
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(feat)
        pred_idx = np.argmax(probs)
        max_prob = np.max(probs)
    else:
        pred = model.predict(feat)
        pred_idx = pred[0]
        max_prob = 0.5 
    if max_prob < threshold:
        label = "Unknown"
    else:
        if hasattr(model, 'classes_'):
            class_id = model.classes_[pred_idx]
        else:
            class_id = pred_idx
        label = CLASSES.get(int(class_id), "Unknown")
        
    print(f"Pred: {label:12s} (confidence: {max_prob:.4f})")
    return label, max_prob, (x1, y1, x2, y2)
    

def main():
    print(f"\n{'='*70}")
    print(f"Material Classification System - Real-Time Inference")
    print(f"{'='*70}\n")
    print(f"Loading model from: Model")
    model = load_inference_model("models/svm_model.pkl")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Camera opened successfully.")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        label, conf, (x1, y1, x2, y2) = predict_frame(model, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} ({conf:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness = 2
        (text_width, text_height) = cv2.getTextSize(text, font, font_scale, thickness)[0]
        cv2.rectangle(frame, 
                     (x1, y1 - text_height - 10),
                     (x1 + text_width + 10, y1),
                     (0, 255, 0), -1)
        cv2.putText(frame, text, 
                   (x1 + 5, y1 - 5),
                   font, font_scale, (0, 0, 0), thickness)
        cv2.imshow('Material Stream Identification System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"\n✓ Processed {frame_count} frames. Exiting...")
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
