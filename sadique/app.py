from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
import cv2
import json
from models import SignLanguageModel
from translation import TranslationModel
import warnings
import os
from werkzeug.utils import secure_filename
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable warnings
warnings.filterwarnings("ignore")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Initialize translation model
try:
    translator = TranslationModel()
    logger.info("Translation model loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load translation model: {str(e)}")
    raise

# Initialize sign language model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

try:
    # Load label map
    with open("label_map.json") as f:
        label_map = json.load(f)
    inv_label_map = {v: k for k, v in label_map.items()}
    
    # Initialize model architecture
    sign_model = SignLanguageModel(num_classes=39).to(device)
    
    # Load model weights
    checkpoint = torch.load("cnn_transformer_sign_model.pth", map_location=device)
    sign_model.load_state_dict(checkpoint['model_state_dict'])
    sign_model.eval()
    logger.info("Sign language model loaded successfully!")
    
except Exception as e:
    logger.error(f"Error loading sign model: {str(e)}")
    inv_label_map = {}

def preprocess_frames(frames):
    """Convert list of frames to model input tensor"""
    try:
        # Ensure we have exactly 16 frames (pad if needed)
        if len(frames) < 16:
            frames += [frames[-1]] * (16 - len(frames))
        elif len(frames) > 16:
            frames = frames[:16]
            
        processed_frames = []
        for frame in frames:
            if frame is None:
                frame = np.zeros((64, 64, 3), dtype=np.uint8)
            frame = cv2.resize(frame, (64, 64))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            processed_frames.append(frame)
            
        frames_array = np.array(processed_frames, dtype=np.float32)
        frames_array = (frames_array - np.mean(frames_array)) / (np.std(frames_array) + 1e-7)
        return torch.from_numpy(frames_array).permute(0, 3, 1, 2)  # (T, C, H, W)
    except Exception as e:
        logger.error(f"Frame processing error: {str(e)}")
        raise

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate():
    try:
        # Get input data
        input_text = request.form.get("text", "").strip()
        target_lang = request.form.get("target_lang", "assamese")
        frames = []
        
        # Process uploaded frames if any
        for key in request.files:
            if key.startswith('frame_'):
                file = request.files[key]
                if file.filename == '':
                    continue
                
                try:
                    img_array = np.frombuffer(file.read(), np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img is not None:
                        frames.append(img)
                    else:
                        logger.warning(f"Failed to decode image: {file.filename}")
                except Exception as e:
                    logger.error(f"Error processing image {file.filename}: {str(e)}")
        
        # Prepare output
        output = {
            "english": input_text,
            target_lang: "",
            "gloss": "",
            "status": "success"
        }

        # Text translation
        if input_text:
            translated = translator.translate(input_text, target_lang=target_lang)
            output[target_lang] = translated if translated else "Translation failed"
        
        # Sign language prediction
        if frames:
            try:
                frames_tensor = preprocess_frames(frames).unsqueeze(0).to(device)
                with torch.no_grad():
                    preds = sign_model(frames_tensor)
                    pred_idx = torch.argmax(preds, dim=1).item()
                output["gloss"] = inv_label_map.get(pred_idx, "Unknown")
                output["confidence"] = float(torch.softmax(preds, dim=1)[0][pred_idx].item())
            except Exception as e:
                logger.error(f"Sign prediction error: {str(e)}")
                output["gloss"] = "Prediction failed"
        
        return jsonify(output)
    
    except Exception as e:
        logger.error(f"Translation endpoint error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)