# IndicSignAI - Indian Sign Language Recognition & Translation

A real-time Indian Sign Language (ISL) recognition and translation web application supporting 13 regional languages.

## 🌟 Features

- **Real-time Sign Detection**: 178+ ISL signs recognized using deep learning
- **Multi-language Translation**: Translate to 13 regional languages
  - Assamese, Hindi, Manipuri, Nepali, Marathi, Odia, Mizorami
  - Gujarati, Tamil, Telugu, Bengali, Meitei Lon, Dzongkha
- **Voice Input**: Speech-to-text for text-to-sign conversion
- **Text-to-Speech**: Audio output for detected signs
- **Web Interface**: Responsive design for desktop and mobile

## 📁 Project Structure

```
PROTOTYPE 5/
├── src/                    # Main application source
│   ├── app.py             # Flask web application
│   └── models/            # ML model modules
│       ├── translation.py
│       ├── isl_image_model.py
│       └── meitei_lon_fallback.py
├── models/                # Trained model files
│   └── ISL_IMAGE.keras/   # Primary ISL model
├── templates/             # HTML templates
├── static/                # CSS, JavaScript
├── training/              # Training scripts & data
│   ├── scripts/
│   └── data/
├── scripts/               # Utility scripts
│   ├── run.bat
│   └── install_dependencies.py
├── archive/               # Legacy files (preserved)
│   ├── legacy_models/
│   ├── experiments/
│   └── prototypes/
├── assets/                # 3D characters, images
└── requirements.txt       # Python dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Using the provided script
python scripts/install_dependencies.py

# Or manually
pip install -r requirements.txt
```

### 2. Run the Application

```bash
# Using the batch script (Windows)
scripts\run.bat

# Or directly
cd src
python app.py
```

### 3. Open in Browser

Navigate to: http://localhost:5000

## 🎯 Usage

1. **Sign to Text**: Show hand signs to the camera
2. **Text to Sign**: Type or speak text to see sign representations
3. **Translation**: Convert between English and regional languages
4. **Voice Input**: Use microphone for speech-to-sign

## 🛠️ System Requirements

- Python 3.8+
- Webcam (for sign detection)
- 4GB+ RAM recommended
- Windows/Linux/macOS

## 📦 Dependencies

- TensorFlow 2.13.0
- OpenCV 4.8.1
- Flask 2.3.3
- MediaPipe 0.10.21
- PyTorch 2.0.1
- Transformers 4.30.2

## 🔧 Configuration

Key settings in `src/app.py`:
- `PROCESS_EVERY_N_FRAMES = 3` - Detection frequency (higher = less CPU)
- `CAMERA_WIDTH = 480` - Camera resolution
- `CONFIDENCE_THRESHOLD = 0.15` - Detection sensitivity

## 📝 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- NLLB-200 for translation capabilities
- MediaPipe for hand tracking
- TensorFlow for deep learning framework
