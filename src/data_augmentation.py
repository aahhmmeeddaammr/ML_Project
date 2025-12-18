import os
import cv2
import random

def augment_image(image):
    h, w = image.shape[:2]
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
    angle = random.uniform(-10, 10)
    scale = random.uniform(0.9, 1.1)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
    image = cv2.warpAffine(
        image, M, (w, h),
        borderMode=cv2.BORDER_REFLECT
    )
    beta = random.uniform(-30, 30)
    image = cv2.convertScaleAbs(image, alpha=1.0, beta=beta)
    return image
    
def load_dataset(data_dir, augment=True):
    images = []
    labels = []
    
    if not os.path.exists(data_dir):
        print(f"Warning: Data directory {data_dir} does not exist.")
        return [], []

    valid_classes = ['0', '1', '2', '3', '4', '5', '6']
    
    for subdir in os.listdir(data_dir):
        class_id_str = subdir.split('_')[0]
        if class_id_str not in valid_classes:
            continue
        
        class_id = int(class_id_str)
        class_path = os.path.join(data_dir, subdir)
        if not os.path.isdir(class_path):
            continue
            
        print(f"Loading class {subdir}...")
        for filename in os.listdir(class_path):
             if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                filepath = os.path.join(class_path, filename)
                img = cv2.imread(filepath)
                if img is not None:
                    images.append(img)
                    labels.append(class_id)
      
    final_images = []
    final_labels = []
    
    unique_classes = set(labels)
    
    for cls in unique_classes:
        cls_indices = [i for i, x in enumerate(labels) if x == cls]
        cls_imgs = [images[i] for i in cls_indices]
        N = len(cls_imgs)
        final_images.extend(cls_imgs)
        final_labels.extend([cls] * N)
        if augment and cls != 6:
            target_augment = int(N * 0.3)
            print(f"  Class {cls} (N={N}): Generating {target_augment} augmented samples (30%)...")
            for _ in range(target_augment):
                src_img = random.choice(cls_imgs)
                aug_img = augment_image(src_img)
                final_images.append(aug_img)
                final_labels.append(cls)
    return final_images, final_labels
