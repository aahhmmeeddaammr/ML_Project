import os
from PIL import Image

def check_images(data_dir):
    print(f"Scanning {data_dir} for corrupted images...")
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                filepath = os.path.join(root, file)
                try:
                    with Image.open(filepath) as img:
                        img.verify()
                except (IOError, SyntaxError) as e:
                    print(f"Corrupted image found: {filepath} - {e}")
                    try:
                        os.remove(filepath)
                        print(f"Deleted: {filepath}")
                    except OSError as e:
                        print(f"Error deleting {filepath}: {e}")
if __name__ == "__main__":
    check_images("data")
