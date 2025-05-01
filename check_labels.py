import os
from char_map import char_to_idx

def check_labels(label_files):
    unknown_chars = set()
    for label_file in label_files:
        if not os.path.exists(label_file):
            print(f"[ERROR] {label_file} does not exist.")
            continue
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    _, label = line.strip().split('\t', 1)
                    for char in label:
                        if char not in char_to_idx:
                            unknown_chars.add(char)
                except ValueError:
                    print(f"[WARNING] Malformed line in {label_file}: {line.strip()}")
    if unknown_chars:
        print(f"Unknown characters: {''.join(sorted(unknown_chars))}")
    else:
        print("All characters in labels are in char_to_idx.")

label_files = [
    'data/train/labels/train_gt.txt',
    'data/val/labels/val_gt.txt',
    'data/test/labels/test_gt.txt'
]
check_labels(label_files)