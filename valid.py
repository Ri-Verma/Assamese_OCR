import os
from PIL import Image

def check_dataset(img_dir):
    print(f"\nChecking dataset at: {img_dir}")
    
    # Count files
    all_files = os.listdir(img_dir)
    images = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    txt_files = [f for f in all_files if f.endswith('.txt')]
    
    print(f"Found {len(images)} images and {len(txt_files)} text files")

    # Check first 5 image-label pairs
    print("\nSample checks:")
    for img in images[:5]:
        base = os.path.splitext(img)[0]
        txt_path = os.path.join(img_dir, base + '.txt')
        
        print(f"\nImage: {img}")
        print(f"Expected label: {txt_path}")
        
        # Check if exists
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                print(f"Label content: '{f.read().strip()}'")
        else:
            print("❌ Label file missing")

# Check your paths
check_dataset('data/train/images')
check_dataset('data/val/images')  # If you have validation data