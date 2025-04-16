import os
import torch
from torch.utils.data import DataLoader
from PIL import Image
from char_map import char_to_idx
from char_map import idx_to_char
from torchvision import transforms
from dataset import AssameseOCRDataset, collate_fn
from model import CRNN
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

#fix device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device: {device}")

#image transforms

transforms = transforms.Compose([
    transforms.Resize((3,100)),transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])


#initialize dataset and dataloaders

train_dataset = AssameseOCRDataset(img_dir='data/train/images',label_file= 'data/train/labels/train_gt.txt', char_to_idx=char_to_idx, transform=transforms)
test_dataset = AssameseOCRDataset(img_dir='data/test/images',label_file='data/test/labels/test_gt.txt', char_to_idx=char_to_idx, transform=transforms)

#print the length of the dataset
print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(test_dataset)}")

# load the data

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,collate_fn=collate_fn)

# set up the model

num_classes = len(char_to_idx) + 1
model = CRNN(img_height=32, nn_classes=num_classes)
model = model.to(device)

# set up training

criterion = nn.CTCLoss(blank=len(char_to_idx), reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=3, factor=0.5, verbose=True
)


#Train

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    batches = 0

    
    for batch in loader:
        if batch is None:
            continue 
    
    for images, labels, input_lengths, target_lengths in loader:
        images = images.to(device)
        labels = labels.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = target_lengths.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, labels, input_lengths, target_lengths)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        
        optimizer.step()
        total_loss += loss.item()
        batches += 1
        
        if batches % 10 == 0:
            print(f"Batch {batches}/{len(loader)}, Loss: {loss.item():.4f}")
    
    return total_loss / batches

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    batches = 0
    
    with torch.no_grad():
        for images, labels, input_lengths, target_lengths in loader:
            images = images.to(device)
            labels = labels.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels, input_lengths, target_lengths)
            
            total_loss += loss.item()
            batches += 1
    
    return total_loss / batches

def decode_predictions(outputs, idx_to_char):
    """Convert model outputs to text"""
    # outputs shape: [seq_len, batch_size, num_classes]
    _, batch_size, _ = outputs.size()
    
    # Get best prediction
    outputs = outputs.permute(1, 0, 2)  # [batch, seq_len, num_classes]
    pred_indices = torch.argmax(outputs, dim=2)  # [batch, seq_len]
    
    decoded_text = []
    for i in range(batch_size):
        indices = pred_indices[i].cpu().numpy()
        
        # Remove repeated characters
        collapsed = []
        for j, ind in enumerate(indices):
            if j == 0 or ind != indices[j-1]:
                collapsed.append(ind)
        
        # Remove blanks
        blank_idx = len(idx_to_char)
        filtered = [idx for idx in collapsed if idx != blank_idx]
        
        # Convert to text
        text = ''.join([idx_to_char[idx] for idx in filtered])
        decoded_text.append(text)
    
    return decoded_text

# Training loop with early stopping
num_epochs = 50
best_val_loss = float('inf')
patience = 7
patience_counter = 0
train_losses = []
val_losses = []

os.makedirs("checkpoints", exist_ok=True)

print("Starting training...")

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = validate(model, test_loader, criterion, device)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    # Update learning rate based on validation loss
    scheduler.step(val_loss)
    
    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, "checkpoints/assamese_ocr_best.pth")
        print(f"Model saved with validation loss: {val_loss:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break

# Save the final model
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': val_loss,
}, "checkpoints/assamese_ocr_final.pth")

# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('loss_plot.png')
plt.show()

# Test on a few examples
model.eval()
with torch.no_grad():
    for i, (images, labels, input_lengths, target_lengths) in enumerate(test_loader):
        if i >= 3:  # Just show first 3 batches
            break
            
        images = images.to(device)
        outputs = model(images)
        
        # Decode predictions
        predictions = decode_predictions(outputs, idx_to_char)
        
        # Decode true labels
        true_texts = []
        label_offset = 0
        for length in target_lengths:
            label = labels[label_offset:label_offset+length].cpu().numpy()
            text = ''.join([idx_to_char[idx] for idx in label])
            true_texts.append(text)
            label_offset += length
        
        # Print results
        for j in range(min(3, len(predictions))):  # First 3 examples in batch
            print(f"Example {i*16+j}:")
            print(f"Predicted: {predictions[j]}")
            print(f"Actual: {true_texts[j]}")
            print("-" * 50)

print("Training complete!")
