import subprocess
import sys

def install_requirements():
    """Install all required dependencies"""
    requirements = [
        "numpy==1.25.2",
        "opencv-python==4.8.1.78", 
        "pandas==2.1.1",
        "Pillow==10.0.1",
        "tensorflow==2.13.0",
        "torch==2.0.1",
        "torchvision==0.15.2", 
        "torchaudio==2.0.2",
        "transformers==4.31.0",
        "sentencepiece==0.1.99",
        "protobuf==4.25.3",
        "mediapipe==0.10.21",
        "attrs==23.2.0",
        "absl-py==2.1.0", 
        "flatbuffers==24.3.25",
        "scipy==1.11.1",
        "Flask==2.3.3",
        "matplotlib==3.7.5"
    ]
    
    print("🚀 Installing dependencies...")
    
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed: {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install: {package}")
    
    print("\n🎉 All dependencies installed successfully!")
    print("👉 Run: python main_system.py")

if __name__ == "__main__":
    install_requirements()