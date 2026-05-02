import torch
from models import SignLanguageModel
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_model():
    try:
        # Initialize model
        model = SignLanguageModel(num_classes=39)
        
        # Create dummy input to initialize all layers
        dummy_input = torch.randn(1, 16, 3, 64, 64)  # batch_size, seq_len, channels, height, width
        model(dummy_input)
        
        # Define label map
        label_map = {
            "all": 0, "bed": 1, "before": 2, "black": 3, "blue": 4, 
            "book": 5, "bowling": 6, "can": 7, "candy": 8, "chair": 9,
            "clothes": 10, "computer": 11, "cool": 12, "cousin": 13, "deaf": 14,
            "dog": 15, "drink": 16, "family": 17, "fine": 18, "finish": 19,
            "fish": 20, "go": 21, "help": 22, "hot": 23, "like": 24,
            "many": 25, "mother": 26, "no": 27, "now": 28, "orange": 29,
            "table": 30, "thanksgiving": 31, "thin": 32, "walk": 33, "what": 34,
            "who": 35, "woman": 36, "year": 37, "yes": 38
        }
        
        # Save model checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'label_map': label_map,
            'config': {
                'num_classes': 39,
                'sequence_length': 16,
                'img_height': 64,
                'img_width': 64,
                'model_architecture': 'CNN-Transformer'
            }
        }, "cnn_transformer_sign_model.pth")
        
        # Save label map separately
        with open("label_map.json", "w") as f:
            json.dump(label_map, f, indent=4)
            
        logger.info("Model and label map saved successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        return False

if __name__ == "__main__":
    if save_model():
        print("Model saved successfully to cnn_transformer_sign_model.pth")
        print("Label map saved to label_map.json")
    else:
        print("Failed to save model")