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

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN(img_height=32, nn_classes=len(idx_to_char) + 1)
model.load_state_dict(torch.load("checkpoints/assamese_ocr.pth", map_location=device))
model.eval()

# Decode function (basic greedy decoder)
def decode(output):
    output = output.permute(1, 0, 2)  # (B, W, C)
    output = output[0]  # Only one sample
    output = torch.argmax(output, dim=1)
    prev = -1
    decoded = []
    for i in output:
        i = i.item()
        if i != prev and i != len(idx_to_char):  # skip blank
            decoded.append(idx_to_char.get(i, ""))
        prev = i
    return ''.join(decoded)

# Prediction function
def ocr(image):
    image = image.convert('L')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)  # (T, B, C)
        output = F.log_softmax(output, dim=2)
    return decode(output)

# Gradio interface
gr.Interface(
    fn=ocr,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Assamese OCR Demo",
    description="Upload a printed Assamese word image to see the prediction."
).launch()
