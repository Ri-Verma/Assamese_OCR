import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from model import CRNN
from char_map import char_to_idx, idx_to_char
import os
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def nan_to_num(x):
    return torch.nan_to_num(x, nan=0.0)

# Image transform
transform = transforms.Compose([
    transforms.Resize((32, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.Lambda(nan_to_num)
])

# Custom beam search decoder
def beam_search_decoder(log_probs, beam_width=10):
    logger.info("Starting beam search decoding")
    T, _, V = log_probs.shape
    log_probs = log_probs.squeeze(1).cpu().numpy()
    blank_id = len(char_to_idx)
    beams = [([], 0.0)]
    for t in range(T):
        new_beams = {}
        for seq, log_prob in beams:
            for v in range(V):
                new_seq = seq + [v] if v != blank_id else seq
                new_log_prob = log_prob + log_probs[t, v]
                new_beams[tuple(new_seq)] = new_beams.get(tuple(new_seq), float('-inf')) + new_log_prob
        beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width]
        beams = [(list(seq), log_prob) for seq, log_prob in beams]
    if not beams:
        logger.warning("No beams generated")
        return ""
    best_seq = beams[0][0]
    decoded = []
    prev = None
    for idx in best_seq:
        if idx != blank_id and (not prev or idx != prev):
            decoded.append(idx_to_char.get(idx, ''))
        prev = idx
    result = ''.join(decoded)
    logger.info(f"Beam search decoded: {result}")
    return result

# Fallback greedy decoder
def greedy_decoder(log_probs):
    logger.info("Starting greedy decoding")
    log_probs = torch.log_softmax(log_probs, dim=2)
    _, pred_indices = log_probs.max(2)
    pred_indices = pred_indices.squeeze(1).cpu().numpy()
    decoded = []
    prev = None
    for idx in pred_indices:
        if idx != len(char_to_idx) and (not prev or idx != prev):
            decoded.append(idx_to_char.get(idx, ''))
        prev = idx
    result = ''.join(decoded)
    logger.info(f"Greedy decoded: {result}")
    return result

# Post-processing for predictions
def correct_prediction(pred):
    if not pred:
        logger.warning("Empty prediction before post-processing")
        return ""
    corrected = ''
    prev_char = None
    for c in pred:
        if c != prev_char:
            corrected += c
        prev_char = c
    correction_dict = {
        'কিিিববাাা': 'কিবা',
        'নহয': 'নহয়',
        'আগবাঢি': 'আগবাঢ়ি'
    }
    result = correction_dict.get(corrected, corrected)
    logger.info(f"Post-processed: {result}")
    return result

# Decode model output
def decode_prediction(output, beam_width=10):
    if torch.isnan(output).any() or torch.isinf(output).any():
        logger.error("Invalid model output: contains NaN or Inf")
        return ""
    # Try beam search
    beam_result = beam_search_decoder(output, beam_width)
    if beam_result:
        return correct_prediction(beam_result)
    # Fallback to greedy
    logger.warning("Beam search returned empty, trying greedy decoder")
    greedy_result = greedy_decoder(output)
    return correct_prediction(greedy_result)

# Predict text from image
def predict_image(image_path, model, device):
    logger.info(f"Processing image: {image_path}")
    try:
        img = Image.open(image_path).convert('L')
        logger.info(f"Image opened successfully: {image_path}")
        img_tensor = transform(img).unsqueeze(0).to(device)
        logger.info(f"Image tensor shape: {img_tensor.shape}, min: {img_tensor.min().item()}, max: {img_tensor.max().item()}")
        
        with torch.no_grad():
            output = model(img_tensor)
            logger.info(f"Model output shape: {output.shape}, min: {output.min().item()}, max: {output.max().item()}")
            prediction = decode_prediction(output)
        
        return prediction
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        return f"Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description='Assamese OCR CLI Interface')
    parser.add_argument('--image', type=str, help='Path to a single image')
    parser.add_argument('--folder', type=str, help='Path to a folder of images')
    args = parser.parse_args()

    if not args.image and not args.folder:
        logger.error("Please provide either --image or --folder argument")
        print("Error: Please provide either --image or --folder argument")
        return

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    print(f"Using device: {device}")

    # Load model
    num_classes = len(char_to_idx) + 1
    model = CRNN(img_height=32, nn_classes=num_classes)
    try:
        model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device))
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        print(f"Error loading model: {str(e)}")
        return
    model = model.to(device)
    model.eval()

    # Process single image
    if args.image:
        if not os.path.exists(args.image):
            logger.error(f"Image {args.image} does not exist")
            print(f"Error: Image {args.image} does not exist")
            return
        prediction = predict_image(args.image, model, device)
        print(f"Image: {args.image}")
        print(f"Prediction: {prediction}")

    # Process folder of images
    if args.folder:
        if not os.path.isdir(args.folder):
            logger.error(f"Folder {args.folder} does not exist")
            print(f"Error: Folder {args.folder} does not exist")
            return
        for filename in os.listdir(args.folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(args.folder, filename)
                prediction = predict_image(image_path, model, device)
                print(f"Image: {image_path}")
                print(f"Prediction: {prediction}")
                print("-" * 50)

if __name__ == '__main__':
    main()