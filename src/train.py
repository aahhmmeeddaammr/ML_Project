import os
import sys
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import data_augmentation, feature_extraction 
DATA_PATH="data"
OUTPUT_DIR='models'
def train_models():
    def preprocess_image(img):
        img = cv2.resize(img, (224, 224))
        if len(img.shape) == 3:
            img = cv2.fastNlMeansDenoisingColored(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
        else:
            img = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
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
        
    def balance_dataset(X, y):
        X_balanced = []
        y_balanced = []
        unique_classes = np.unique(y)
        for cls in unique_classes:
            idx = np.where(y == cls)[0]
            X_cls = X[idx]
            y_cls = y[idx]
            replace = len(X_cls) < 500
            X_resampled, y_resampled = resample(X_cls, y_cls, 
                                                replace=replace, 
                                                n_samples=500, 
                                                random_state=42)
            X_balanced.append(X_resampled)
            y_balanced.append(y_resampled)            
        return np.concatenate(X_balanced), np.concatenate(y_balanced)

    images, labels = data_augmentation.load_dataset(DATA_PATH, augment=True)
    
    X_raw = []
    y_raw = []
    
    for i, img in enumerate(images):
        processed_img = preprocess_image(img)
        feature = feature_extraction.extract_features(processed_img)
        X_raw.append(feature)
        y_raw.append(labels[i])
            
    X = np.array(X_raw)
    y = np.array(y_raw)
    
    X, y = balance_dataset(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
   
    knn_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(metric='euclidean', weights='distance'))
    ])
   
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            kernel='rbf',
            C=100,
            gamma=0.001,
            class_weight='balanced',
            probability=True,
            random_state=42
        ))
    ])
    
    knn_params = {
        'knn__n_neighbors': [3],
        'knn__weights': ['distance'],
        'knn__metric': ['euclidean']
    }
    
    svm_params = {
        'svm__C': [10],
        'svm__kernel': ['rbf'],
        'svm__gamma': [ 0.001,],
        'svm__degree': [2]
    }
    
    knn_search = GridSearchCV(knn_pipeline, knn_params, cv=5, n_jobs=-1, verbose=2, scoring='accuracy') 
    knn_search.fit(X_train, y_train)
    best_knn = knn_search.best_estimator_
    
    knn_pred = best_knn.predict(X_test)
    knn_acc = accuracy_score(y_test, knn_pred)
    print(f"k-NN Test Accuracy: {knn_acc:.4f}")
    print(classification_report(y_test, knn_pred))
    
    joblib.dump(best_knn, os.path.join(OUTPUT_DIR, 'knn_model.pkl'))
    print("Saved k-NN model.")
    print("Training SVM...")
    svm_search = GridSearchCV(svm_pipeline, svm_params, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
    svm_search.fit(X_train, y_train)
    best_svm = svm_search.best_estimator_
    print(f"Best SVM params: {svm_search.best_params_}")
    print(f"Best SVM CV score: {svm_search.best_score_:.4f}")
    
    svm_pred = best_svm.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_pred)
    print(f"SVM Test Accuracy: {svm_acc:.4f}")
    print("SVM Performance:")
    print(classification_report(y_test, svm_pred))
    
    joblib.dump(best_svm, os.path.join(OUTPUT_DIR, 'svm_model.pkl'))
    print("✓ Saved SVM model.")
    print("=== Training Complete ===")

if __name__ == "__main__":
    train_models()
