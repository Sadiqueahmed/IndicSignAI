from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
import cv2
import json
from models import SignLanguageModel
from translation import TranslationModel
import warnings
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Resolve library conflicts

warnings.filterwarnings("ignore")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings


app = Flask(__name__)

# Initialize translation model
translator = TranslationModel()

# Load PyTorch sign language model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device set to use {device}")

# Initialize model architecture
sign_model = SignLanguageModel(num_classes=39).to(device)

try:
    # Load checkpoint
    checkpoint = torch.load("cnn_transformer_sign_model.pth", map_location=device)
    sign_model.load_state_dict(checkpoint['model_state_dict'])
    sign_model.eval()
    print("Sign language model loaded successfully!")
    
    # Load label map
    with open("label_map.json") as f:
        label_map = json.load(f)
    inv_label_map = {v: k for k, v in label_map.items()}
    
except Exception as e:
    print(f"Error loading sign model: {str(e)}")
    inv_label_map = {}

def preprocess_frames(frames):
    """Convert list of frames to model input tensor"""
    frames = [cv2.resize(frame, (64, 64)) for frame in frames]
    frames = np.array(frames, dtype=np.float32)
    frames = (frames - np.mean(frames)) / (np.std(frames) + 1e-7)
    return torch.from_numpy(frames).permute(0, 3, 1, 2)  # (T, C, H, W)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    input_text = data.get("text", "")
    output = {"english": "", "assamese": "", "gloss": ""}

    # Text translation
    if input_text:
        translated_text = translator.translate(input_text)
        output["english"] = input_text
        output["assamese"] = translated_text if translated_text else "Translation failed"

    # Sign language prediction (simulated)
    try:
        dummy_frames = [np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(16)]
        frames_tensor = preprocess_frames(dummy_frames).unsqueeze(0).to(device)
        
        with torch.no_grad():
            preds = sign_model(frames_tensor)
            pred_idx = torch.argmax(preds, dim=1).item()
        
        output["gloss"] = inv_label_map.get(pred_idx, "Unknown")
    except Exception as e:
        print(f"Sign prediction error: {str(e)}")
        output["gloss"] = "Prediction failed"

    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=False, port=5000)