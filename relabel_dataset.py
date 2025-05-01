import os
from char_map import char_to_idx

def relabel_dataset(label_file, output_file):
    if not os.path.exists(label_file):
        print(f"[ERROR] {label_file} does not exist.")
        return
    with open(label_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    valid_lines = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in lines:
            try:
                img_name, label = line.strip().split('\t', 1)
                valid_label = True
                for char in label:
                    if char not in char_to_idx:
                        print(f"[ERROR] Unknown character '{char}' in {img_name}, label: {label}")
                        valid_label = False
                        break
                if valid_label:
                    f.write(f"{img_name}\t{label}\n")
                    valid_lines += 1
            except ValueError:
                print(f"[WARNING] Malformed line in {label_file}: {line.strip()}")
    print(f"Generated {output_file} with {valid_lines}/{len(lines)} lines.")

label_files = [
    ('data/train/labels/train_gt.txt', 'data/train/labels/train_gt_new.txt'),
    ('data/val/labels/val_gt.txt', 'data/val/labels/val_gt_new.txt'),
    ('data/test/labels/test_gt.txt', 'data/test/labels/test_gt_new.txt')
]
for input_file, output_file in label_files:
    relabel_dataset(input_file, output_file)