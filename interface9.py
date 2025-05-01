import gradio as gr
import torch
from PIL import Image
from torchvision import transforms
from model import CRNN
from char_map import idx_to_char
import torch.nn.functional as F

# Set up transform
transform = transforms.Compose([
    transforms.Resize((32, 100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = CRNN(img_height=32, nn_classes=60+ 1).to(device)

# Safe model loading
try:
    model.load_state_dict(torch.load("checkpoints/assamese_ocr.pth", map_location=device))
    model.eval()
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    exit()

# Decode function (greedy CTC decoder)
def decode(output):
    output = output.permute(1, 0, 2)  # (B, W, C)
    output = output[0]  # Only one sample
    output = torch.argmax(output, dim=1)
    prev = -1
    decoded = []
    for i in output:
        i = i.item()
        if i != prev and i != len(idx_to_char):  # Skip blank
            decoded.append(idx_to_char.get(i, ""))
        prev = i
    return ''.join(decoded)

# OCR function
def ocr(image):
    try:
        image = image.convert('L')
        image = transform(image).unsqueeze(0).to(device)  # Move to device
        with torch.no_grad():
            output = model(image)  # (T, B, C)
            output = F.log_softmax(output, dim=2)
        return decode(output)
    except Exception as e:
        print(f"[ERROR] during OCR: {e}")
        return "Error processing image."

# Gradio interface using Blocks
with gr.Blocks() as demo:
    gr.Markdown("## Assamese OCR Demo")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload image")
        with gr.Column():
            output_text = gr.Textbox(label="Predicted Text")
    image_input.change(fn=ocr, inputs=image_input, outputs=output_text)

demo.launch()
