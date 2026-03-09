from PIL import Image
from pathlib import Path

# Directory containing images
image_dir = Path("merged_gauges_csv")

def resize_longest_side(image, target_size=720):
    w, h = image.size

    if w > h:
        new_w = target_size
        new_h = int(h * target_size / w)
    else:
        new_h = target_size
        new_w = int(w * target_size / h)

    return image.resize((new_w, new_h), Image.BICUBIC)

# Supported image extensions
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for img_path in image_dir.iterdir():
    if img_path.suffix.lower() in extensions:
        try:
            with Image.open(img_path) as img:
                resized = resize_longest_side(img, 720)
                resized.save(img_path)  # overwrite with same filename
                print(f"Resized: {img_path.name}")
        except Exception as e:
            print(f"Skipped {img_path.name}: {e}")

