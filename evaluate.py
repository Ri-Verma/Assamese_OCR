import torch
from torch.utils.data import DataLoader
from char_map import char_to_idx, idx_to_char
from dataset import AssameseOCRDataset, collate_fn
from model import CRNN
from torchvision import transforms
import editdistance

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def decode_predictions(outputs, idx_to_char):
    """Decode CTC outputs to text using greedy decoding."""
    outputs = outputs.permute(1, 0, 2)  # [batch, seq_len, classes]
    _, max_indices = torch.max(outputs, dim=2)  # [batch, seq_len]
    decoded = []
    for batch_idx in range(max_indices.size(0)):
        seq = max_indices[batch_idx].tolist()
        text = []
        prev = None
        for idx in seq:
            if idx != prev and idx != len(char_to_idx):  # Skip blank and repeats
                text.append(idx_to_char.get(idx, ''))
            prev = idx
        decoded.append(''.join(text))
    return decoded

def compute_wer_cer(preds, targets):
    """Compute WER and CER."""
    wer, cer = 0, 0
    total_words, total_chars = 0, 0
    for pred, target in zip(preds, targets):
        # WER
        pred_words = pred.split()
        target_words = target.split()
        wer += editdistance.eval(pred_words, target_words)
        total_words += len(target_words)
        # CER
        cer += editdistance.eval(pred, target)
        total_chars += len(target)
    wer = wer / total_words if total_words > 0 else float('inf')
    cer = cer / total_chars if total_chars > 0 else float('inf')
    return wer, cer

def evaluate():
    transform = transforms.Compose([
        transforms.Resize((32, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.Lambda(lambda x: torch.nan_to_num(x, nan=0.0))
    ])
    test_dataset = AssameseOCRDataset(
        img_dir='data/test/images',
        label_file='data/test/labels/test_gt.txt',
        char_to_idx=char_to_idx,
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                             collate_fn=collate_fn, num_workers=2)
    model = CRNN(img_height=32, nn_classes=len(char_to_idx) + 1)
    model.load_state_dict(torch.load('checkpoints/best_model.pth'))
    model.to(device)
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
            images, labels, input_lengths, target_lengths = batch
            images = images.to(device)
            outputs = model(images)
            preds = decode_predictions(outputs, idx_to_char)
            targets = []
            for i in range(labels.size(0)):
                target = ''.join(idx_to_char.get(idx.item(), '') for idx in labels[i][:target_lengths[i]])
                targets.append(target)
            all_preds.extend(preds)
            all_targets.extend(targets)
    wer, cer = compute_wer_cer(all_preds, all_targets)
    print(f"Test WER: {wer:.4f}, Test CER: {cer:.4f}")
    return wer, cer

if __name__ == "__main__":
    evaluate()