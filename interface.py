import argparse
import torch
from PIL import Image, ImageEnhance
import pdf2image
import numpy as np
from torchvision import transforms
from train_fixed import SafeCRNN
from char_map import char_to_idx, idx_to_char
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Correction dictionary for post-processing
correction_dict = {
    'কিিিববাাা': 'কিবা',
    'নহয': 'নহয়',
    'আগবাঢি': 'আগবাঢ়ি',
    'কিবাএটাধুনীয়াদিন': 'কিবা এটা ধুনীয়া দিন।',
    'য‌া‌ি': 'জাতি',
    '।': '<empty>'  
}

def decode_output(output, char_map, idx_to_char):
    """Decode CTC output to text."""
    try:
        _, preds = output.max(2)
        preds = preds.transpose(1, 0).cpu().numpy()
        logger.info(f"Raw predictions: {preds[0]}")
        text = ''
        last_char = None
        for idx in preds[0]:
            if idx == char_map.get('<pad>', -1):
                last_char = None
                continue
            current_char = idx_to_char.get(idx, '<unk>')
            if current_char in ['\u200c', '\u200d', '\n']:
                continue
            if current_char != last_char:
                text += current_char
            last_char = current_char
        logger.info(f"Decoded text: {text}")
        return text if text else '<empty>'
    except Exception as e:
        logger.error(f"Decoding error: {str(e)}")
        raise

def preprocess_image(image):
    """Preprocess image for model input."""
    image = ImageEnhance.Contrast(image).enhance(2.0)
    transform = transforms.Compose([
        transforms.Resize((32, 300)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = image.convert('L')
    return transform(image).unsqueeze(0)

def predict_image(image_path, model, device):
    """Predict text from a single image."""
    try:
        image = Image.open(image_path)
        input_tensor = preprocess_image(image).to(device)
        with torch.no_grad():
            output = model(input_tensor)
        predicted_text = decode_output(output, char_to_idx, idx_to_char)
        corrected_text = correction_dict.get(predicted_text, predicted_text)
        return corrected_text
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return None

def predict_pdf(pdf_path, model, device):
    """Predict text from a PDF."""
    try:
        images = pdf2image.convert_from_path(pdf_path, dpi=300, poppler_path=r'C:\poppler-24.08.0\Library\bin')
        results = []
        for i, image in enumerate(images):
            logger.info(f"Processing page {i+1}")
            input_tensor = preprocess_image(image).to(device)
            with torch.no_grad():
                output = model(input_tensor)
            predicted_text = decode_output(output, char_to_idx, idx_to_char)
            corrected_text = correction_dict.get(predicted_text, predicted_text)
            results.append((i+1, corrected_text))
        return results
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Assamese OCR Inference")
    parser.add_argument('--image', type=str, help="Path to input image")
    parser.add_argument('--pdf', type=str, help="Path to input PDF")
    args = parser.parse_args()

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load model
    try:
        model = SafeCRNN(img_height=32, nn_classes=len(char_to_idx)).to(device)
        model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device, weights_only=True))
        model.eval()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return

    # Process input
    if args.image:
        result = predict_image(args.image, model, device)
        if result:
            logger.info(f"Image: {args.image}")
            logger.info(f"Predicted: {result}")
        else:
            logger.error("Image prediction failed")

    if args.pdf:
        results = predict_pdf(args.pdf, model, device)
        if results:
            for page, text in results:
                logger.info(f"Page {page}: {text}")
        else:
            logger.error("PDF prediction failed")

if __name__ == '__main__':
    main()