import os
import torch
from torch.utils.data import DataLoader
from char_map import char_to_idx, idx_to_char
from torchvision import transforms
from dataset import AssameseOCRDataset, collate_fn
from model import CRNN
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Fix device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device: {device}")

# Define named function for transform to avoid pickling issues
def nan_to_num(x):
    return torch.nan_to_num(x, nan=0.0)

# Image transforms
transform = transforms.Compose([
    transforms.Resize((32, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.RandomApply([transforms.RandomChoice([
        transforms.RandomAdjustSharpness(sharpness_factor=0.5),
        transforms.RandomAdjustSharpness(sharpness_factor=2.0)
    ])], p=0.3),
    transforms.Lambda(nan_to_num),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))  
])

# ================== Critical Safety Checks ==================
def validate_char_coverage(label_file):
    all_chars = set()
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                _, label = line.strip().split('\t', 1)
                all_chars.update(label)
            except ValueError:
                print(f"[WARNING] Malformed line in {label_file}: {line.strip()}")
    missing = all_chars - set(char_to_idx.keys())
    if missing:
        print(f"CRITICAL: Missing {len(missing)} characters in char_to_idx:")
        print("Missing characters:", ''.join(sorted(missing)))
        return False
    return True

if not (validate_char_coverage('data/train/labels/train_gt.txt') and
        validate_char_coverage('data/val/labels/val_gt.txt') and
        validate_char_coverage('data/test/labels/test_gt.txt')):
    print("ERROR: Character coverage validation failed. Regenerate char_map.py.")
    exit(1)

# ================== Dataset Verification ==================
class SafeAssameseOCRDataset(AssameseOCRDataset):
    def __init__(self, img_dir, label_file, char_to_idx, transform=None, indices=None, max_images=None):
        super().__init__(img_dir, label_file, char_to_idx, transform, max_images)
        self.char_to_idx = char_to_idx
        self.transform = transform
        self._filter_invalid_samples()
        if indices is not None:
            valid_indices = [i for i in indices if i < len(self.image_files) and i in self.valid_indices]
            self.valid_indices = valid_indices
        print(f"Dataset ({img_dir}): {len(self.valid_indices)}/{len(self.image_files)} valid samples")

    def _filter_invalid_samples(self):
        valid_indices = []
        for idx in range(len(self.image_files)):
            try:
                img, label = super().__getitem__(idx)
                if img is None:
                    print(f"Skipping {self.image_files[idx]}: None image")
                    continue
                if torch.isnan(img).any() or torch.isinf(img).any():
                    print(f"Skipping {self.image_files[idx]}: Invalid image values (nan/inf)")
                    continue
                if label is None:
                    print(f"Skipping {self.image_files[idx]}: None label")
                    continue
                valid_indices.append(idx)
            except Exception as e:
                print(f"Skipping {self.image_files[idx]}: Error - {str(e)}")
                continue
        print(f"Filtered dataset: {len(valid_indices)}/{len(self.image_files)} valid samples")
        self.valid_indices = valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        if idx >= len(self.valid_indices):
            raise IndexError(f"Index {idx} out of range for {len(self.valid_indices)} valid samples")
        actual_idx = self.valid_indices[idx]
        return super().__getitem__(actual_idx)

# ================== Model Setup ==================
class SafeCRNN(CRNN):
    def __init__(self, img_height, nn_classes):
        super(SafeCRNN, self).__init__(img_height, nn_classes)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = super().forward(x)
        return torch.clamp(x, min=-100, max=100)

# ================== CTC Loss Wrapper ==================
class SafeCTCLoss(nn.Module):
    def __init__(self, blank=0, reduction='mean'):
        super().__init__()
        self.blank = blank
        self.reduction = reduction
        self.ctc = nn.CTCLoss(blank=blank, reduction='none', zero_infinity=True)
        
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        print(f"CTCLoss - log_probs shape: {log_probs.shape}, targets shape: {targets.shape}")
        print(f"Input lengths: {input_lengths}, Target lengths: {target_lengths}")
        print(f"Log probs stats - Min: {log_probs.min().item()}, Max: {log_probs.max().item()}, Nan: {torch.isnan(log_probs).any()}, Inf: {torch.isinf(log_probs).any()}")
        print(f"Targets range - Min: {targets.min().item()}, Max: {targets.max().item()}")
        mask = (target_lengths <= input_lengths) & (input_lengths > 0) & (targets.max() <= len(char_to_idx))
        print(f"Mask: {mask}, Valid samples: {mask.sum().item()}")
        if not mask.any():
            print("CTCLoss: No valid samples, returning 0.0")
            return torch.tensor(0.0).to(log_probs.device)
        log_probs = log_probs[:, mask]
        targets = targets[mask]
        input_lengths = input_lengths[mask]
        target_lengths = target_lengths[mask]
        if log_probs.size(1) == 0:
            print("CTCLoss: Empty batch after masking, returning 0.0")
            return torch.tensor(0.0).to(log_probs.device)
        loss = self.ctc(log_probs, targets, input_lengths, target_lengths)
        print(f"Raw CTC loss: {loss}")
        print(f"Raw CTC loss stats - Min: {loss.min().item()}, Max: {loss.max().item()}, Nan: {torch.isnan(loss).any()}")
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"Processed CTC loss: {loss}")
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# ================== Training Setup ==================
def main():
    # Verify char_to_idx
    print(f"Number of characters in char_to_idx: {len(char_to_idx)}")

    # Load datasets
    train_dataset = SafeAssameseOCRDataset(
        img_dir='data/train/images',
        label_file='data/train/labels/train_gt.txt',
        char_to_idx=char_to_idx,
        transform=transform,
        max_images=79000
    )
    val_dataset = SafeAssameseOCRDataset(
        img_dir='data/val/images',
        label_file='data/val/labels/val_gt.txt',
        char_to_idx=char_to_idx,
        transform=transform
    )
    test_dataset = SafeAssameseOCRDataset(
        img_dir='data/test/images',
        label_file='data/test/labels/test_gt.txt',
        char_to_idx=char_to_idx,
        transform=transform
    )
    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}, Test dataset size: {len(test_dataset)}")

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, 
                             collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False,
                            collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                             collate_fn=collate_fn, num_workers=2)
    print(f"Train loader batches: {len(train_loader)}, Validation loader batches: {len(val_loader)}")

    # Model
    num_classes = len(char_to_idx) + 1
    model = SafeCRNN(img_height=32, nn_classes=num_classes)
    model._init_weights()
    model = model.to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.01)
    criterion = SafeCTCLoss(blank=len(char_to_idx), reduction='mean')
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=3, factor=0.5
    )

    # Warmup scheduler
    warmup_epochs = 5
    def lr_lambda(epoch):
        return min(1.0, (epoch + 1) / warmup_epochs)
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Sanity check
    try:
        sample_batch = next(iter(train_loader))
        if sample_batch is not None:
            images, labels, input_lengths, target_lengths = sample_batch
            print("\n=== Sanity Check ===")
            print("Image stats - Min:", images.min().item(), "Max:", images.max().item(), "Std:", images.std().item())
            print("Label lengths:", target_lengths.numpy())
            print("Input lengths:", input_lengths.numpy())
            print("Labels:", labels)
            with torch.no_grad():
                sample_output = model(images[:1].to(device))
                print("Model output shape:", sample_output.shape)
                print("Model output stats - Min:", sample_output.min().item(),
                      "Max:", sample_output.max().item(),
                      "Nan:", torch.isnan(sample_output).any(),
                      "Inf:", torch.isinf(sample_output).any())
        else:
            print("Sanity check failed: Empty batch")
    except StopIteration:
        print("ERROR: Train loader is empty!")
        return

    # Training loop
    num_epochs = 50
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience = 5
    no_improve_epochs = 0
    
    torch.cuda.empty_cache()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        skipped_batches = 0
        
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                skipped_batches += 1
                print(f"Skipped batch {batch_idx} in training")
                continue
            images, labels, input_lengths, target_lengths = batch
            
            images = images.to(device)
            labels = labels.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)
            
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                outputs = model(images)
                print(f"Model outputs - Contains nan: {torch.isnan(outputs).any()}, Contains inf: {torch.isinf(outputs).any()}")
                outputs = torch.log_softmax(outputs, 2)
                loss = criterion(outputs, labels, input_lengths, target_lengths)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
                print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

        val_loss = 0
        val_skipped_batches = 0
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch is None:
                    val_skipped_batches += 1
                    print(f"Skipped batch {batch_idx} in validation")
                    continue
                images, labels, input_lengths, target_lengths = batch
                
                images = images.to(device)
                labels = labels.to(device)
                input_lengths = input_lengths.to(device)
                target_lengths = target_lengths.to(device)
                
                outputs = model(images)
                print(f"Validation model outputs - Contains nan: {torch.isnan(outputs).any()}, Contains inf: {torch.isinf(outputs).any()}")
                outputs = torch.log_softmax(outputs, 2)
                loss = criterion(outputs, labels, input_lengths, target_lengths)
                val_loss += loss.item()
        
        train_loss = total_loss / (len(train_loader) - skipped_batches) if len(train_loader) > skipped_batches else float('inf')
        val_loss = val_loss / (len(val_loader) - val_skipped_batches) if len(val_loader) > val_skipped_batches else float('inf')
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Training skipped batches: {skipped_batches}/{len(train_loader)}")
        print(f"Validation skipped batches: {val_skipped_batches}/{len(val_loader)}")
        
        if val_loss < best_val_loss and val_loss != float('inf'):
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
        else:
            no_improve_epochs += 1
        
        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    torch.save(model.state_dict(), "checkpoints/final_model.pth")
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_curve.png')

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  
    main()