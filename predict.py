# predict.py
import os
import torch
from torch.utils.data import DataLoader
from model import CRNN
from dataset import AssameseOCRDataset, collate_fn
from char_map import idx_to_char, char_to_idx
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def decode_prediction(preds, pred_sizes):
    decoded_texts = []
    preds = preds.permute(1, 0, 2).cpu()
    preds = torch.argmax(preds, dim=2)

    for i in range(preds.size(0)):
        pred_seq = preds[i][:pred_sizes[i]]
        prev_char = -1
        text = ''
        for idx in pred_seq:
            idx = idx.item()
            if idx != prev_char and idx != len(char_to_idx):  # Skip repeated and blank
                text += idx_to_char.get(idx, '')
            prev_char = idx
        decoded_texts.append(text)
    return decoded_texts

transform = transforms.Compose([
    transforms.Resize((32, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load dataset
test_dataset = AssameseOCRDataset(
    img_dir='data/test/images',
    label_file='data/test/labels/test_gt.txt',
    char_to_idx=char_to_idx,
    transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# Load model
num_classes = len(char_to_idx) + 1
model = CRNN(img_height=32, nn_classes=num_classes)
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
model.to(device)
model.eval()

# Run inference and save predictions
output_file = "predictions.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    with torch.no_grad():
        for batch in test_loader:
            images, labels, input_lengths, target_lengths = batch
            images = images.to(device)

            outputs = model(images)
            outputs = torch.log_softmax(outputs, 2)
            pred_sizes = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.int32)
            decoded = decode_prediction(outputs, pred_sizes)

            image_name = test_dataset.image_files[test_loader.dataset.valid_indices[test_loader._index]]
            f.write(f"{os.path.basename(image_name)}\t{decoded[0]}\n")

print(f"Saved predictions to {output_file}")
