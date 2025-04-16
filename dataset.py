import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from char_map import char_to_idx
from torchvision import transforms

class AssameseOCRDataset(Dataset):
    def __init__(self, img_dir, label_file, char_to_idx, transform=None):
        self.img_dir = img_dir
        self.char_to_idx = char_to_idx
        self.transform = transform

        # Read labels from central label file
        self.labels = {}
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) != 2:
                        continue
                    filename, text = parts
                    self.labels[filename] = text

        # Keep only image files that have labels
        self.image_files = [f for f in os.listdir(img_dir)
                            if f.endswith('.jpeg') and f in self.labels]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        try:
            img = Image.open(img_path).convert('L')
        except Exception as e:
            print(f"[ERROR] Failed to load image {img_name}: {e}")
            return None

        label = self.labels.get(img_name)
        if label is None:
            print(f"[ERROR] Label not found for image {img_name}")
            return None

        if self.transform:
            img = self.transform(img)

        try:
            label_encoded = [self.char_to_idx[c] for c in label]
        except KeyError as e:
            print(f"[ERROR] Unknown character '{e.args[0]}' in label for {img_name}")
            return None

        return img, torch.tensor(label_encoded, dtype=torch.long)

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)

    target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    labels = torch.cat(labels)

    batch_size, _, _, width = images.size()
    cnn_output_width = width // 4
    input_lengths = torch.full(size=(batch_size,), fill_value=cnn_output_width, dtype=torch.long)

    return images, labels, input_lengths, target_lengths
