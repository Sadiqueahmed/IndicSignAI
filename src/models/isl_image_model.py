"""
ISL Image Model Integration Module
Handles loading and inference for the ISL_IMAGE.keras model (Keras 3.x format)
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import json
import logging
from collections import deque

logger = logging.getLogger(__name__)

# --- Transformer Block Definition (Required for model loading) ---
class TransformerBlock(layers.Layer):
    def __init__(self, dim, heads, ff_dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.heads = heads
        self.ff_dim = ff_dim

        self.att = layers.MultiHeadAttention(
            num_heads=heads,
            key_dim=dim // heads
        )

        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(dim)
        ])

        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()

    def call(self, x):
        attn = self.att(x, x)
        x = self.norm1(x + attn)
        ffn = self.ffn(x)
        return self.norm2(x + ffn)

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "heads": self.heads,
            "ff_dim": self.ff_dim,
        })
        return config

class ISLImageModel:
    """
    Handler for ISL_IMAGE.keras model (Keras 3.x directory format)
    Input: 160x160 RGB images
    Output: 178 ISL sign classes
    """
    
    # Resolve model path relative to THIS file's directory
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
    
    def __init__(self, model_path=None):
        # Resolve to absolute path based on this file's directory
        if model_path is None:
            model_path = os.path.join(self._PROJECT_ROOT, 'ISL_IMAGE.keras')
        elif not os.path.isabs(model_path):
            # Try relative to project root first, then current dir
            candidate = os.path.join(self._PROJECT_ROOT, model_path)
            if os.path.exists(candidate):
                model_path = candidate
            else:
                model_path = os.path.abspath(model_path)

        self.model_path = model_path
        self.model = None
        self.class_names = self._load_class_names()
        self.input_size = (160, 160)
        self.confidence_threshold = 0.65
        
        # Prediction smoothing buffer
        self.prediction_buffer = deque(maxlen=8)
        
        logger.info(f"ISLImageModel initialized with path: {self.model_path}")
        
    def _load_class_names(self):
        """Load ISL class names - 178 classes"""
        return ['A LOT', 'ABUSE', 'ALL', 'ANGRY', 'ANY', 'ANYTHING', 'APPRECIATE',
        'BEAUTIFUL', 'BED', 'BORED', 'BRING', 'CLASS', 'COLD', 'COLLEGE_SCHOOL', 'COMB',
        'COME', 'CRYING', 'DARE', 'DIFFERENCE', 'DILEMMA', 'DISAPPOINTED', 'DO', "DON'T CARE",
        'ENJOY', 'FAVOUR', 'FEVER', 'FINE', 'FOOD', 'FREE', 'FRIEND', 'GLASS', 'GO',
        'GOOD', 'GOT', 'GRATEFUL', 'HAD', 'HAPPENED', 'HAPPY', 'HEAR', 'HEART',
        'HELLO_HI', 'HELP', 'HIDING', 'HOW', 'HURT', 'I_ME_MINE_MY', 'KIND', 'KNOW',
        'LEAVE', 'LIGHT', 'LIKE', 'LIKE_LOVE', 'MAKE', 'MEAN IT', 'MEDICINE', 'NAME',
        'NEED', 'NEVER', 'NICE', 'NOT', 'NOW', 'NUMBER', 'OLD_AGE', 'ON THE WAY',
        'ONWARDS', 'OUTSIDE', 'PHONE', 'PLACE', 'PLANNED', 'POUR', 'PREPARE', 'PROMISE',
        'REALLY', 'REPEAT', 'ROOM', 'SERVE', 'SHIRT', 'SITTING', 'SLEEP', 'SLOWER',
        'SO MUCH', 'SOFTLY', 'SOME HOW', 'SOME MORE', 'SOME ONE', 'SOMETHING', 'SORRY',
        'SPEAK', 'STUBBORN', 'SURE', 'TAKE CARE', 'TAKE TIME', 'TALK', 'TELL', 'THANK',
        'THAT', 'THERE', 'THINGS', 'THINK', 'THIS ONE', 'TIRED', 'TRAIN', 'TRUST',
        'TRUTH', 'TURN ON', 'VERY', 'WANT', 'WATER', 'WEAR', 'WELCOME', 'WHAT', 'WHEN',
        'WHO', 'WORRY', 'afraid', 'again', 'agree', 'answer', 'assistance', 'attendance',
        'bad', 'become', 'book', 'break', 'careful', 'change', 'chat', 'college',
        'congratulations', 'doctor', 'email', 'file', 'from', 'good morning',
        'happy birthday', 'home', 'how are you', 'hungry', 'i need help', 'join',
        'keepsmile', 'meet', 'mistake', 'open', 'opinion', 'pain', 'pass', 'please',
        'practice', 'pray', 'pressure', 'problem', 'questions', 'remember', 'seat',
        'secondary', 'shift', 'sick', 'skin', 'small', 'specific', 'stand', 'stop',
        'sun', 'team', 'thirsty', 'this', 'today', 'together', 'understand', 'wait',
        'warn', 'where', 'which', 'work', 'write', 'you']
    
    def _build_model(self):
        """Build the model architecture from scratch"""
        inputs = keras.Input(shape=(160, 160, 3))
        
        # MobileNetV2 Backbone (frozen)
        base_model = keras.applications.MobileNetV2(
            input_shape=(160, 160, 3), 
            include_top=False, 
            weights=None
        )
        x = base_model(inputs)
        
        # Reshape for sequence processing
        x = layers.Reshape((25, 1280))(x)
        
        # Transformer Block
        x = TransformerBlock(dim=1280, heads=4, ff_dim=512)(x)
        
        # Classification head
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(178, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        return model
    
    def load_model(self):
        """Load the ISL_IMAGE.keras model (Keras 3.x directory format)"""
        try:
            # Verify path exists before attempting to load
            if not os.path.exists(self.model_path):
                logger.error(f"FILE NOT FOUND AT PATH: {self.model_path}")
                print(f"[X] FILE NOT FOUND AT PATH: {self.model_path}")
                # Try alternate locations
                alt_paths = [
                    os.path.join(self._PROJECT_ROOT, 'ISL_IMAGE.keras'),
                    os.path.join(self._PROJECT_ROOT, 'models', 'ISL_IMAGE.keras'),
                    os.path.join(self._THIS_DIR, '..', '..', 'ISL_IMAGE.keras'),
                ]
                for alt in alt_paths:
                    abs_alt = os.path.abspath(alt)
                    if os.path.exists(abs_alt):
                        self.model_path = abs_alt
                        print(f"  [RESOLVED] Found model at alternate path: {abs_alt}")
                        break
                    else:
                        print(f"  FILE NOT FOUND AT PATH: {abs_alt}")
                else:
                    return False

            # Check if it's a directory (Keras 3.x format)
            if os.path.isdir(self.model_path):
                logger.info(f"Loading ISL Image Model from directory: {self.model_path}")
                
                # Build model architecture
                self.model = self._build_model()
                
                # Load weights
                weights_path = os.path.join(self.model_path, 'model.weights.h5')
                if os.path.exists(weights_path):
                    try:
                        self.model.load_weights(weights_path)
                        logger.info("✓ Weights loaded successfully")
                    except Exception as w_err:
                        logger.warning(f"Weight loading error: {w_err}")
                        logger.info("Attempting to load with skip_mismatch...")
                        self.model.load_weights(weights_path, skip_mismatch=True)
                        logger.info("✓ Weights loaded with skip_mismatch")
                else:
                    logger.error(f"Weights file not found: {weights_path}")
                    return False

                
            else:
                # Try loading as a single file (older format)
                logger.info(f"Loading ISL Image Model from file: {self.model_path}")
                
                custom_objects = {'TransformerBlock': TransformerBlock}
                
                try:
                    self.model = keras.models.load_model(
                        self.model_path, 
                        custom_objects=custom_objects,
                        compile=False
                    )
                    logger.info("✓ Model loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load model: {e}")
                    return False
            
            # Compile model
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info(f"✓ Model ready - Input shape: {self.model.input_shape}, Output classes: {len(self.class_names)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ISL Image Model: {e}")
            return False
    
    def preprocess_image(self, image):
        """
        Preprocess image for model inference
        Following the notebook approach exactly
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Resize to model input size
        image_resized = cv2.resize(image_rgb, self.input_size)
        
        # Expand dimensions for batch
        image_batch = np.expand_dims(image_resized, axis=0)
        
        # Apply MobileNetV2 preprocessing (VERY IMPORTANT as per notebook)
        image_preprocessed = preprocess_input(image_batch)
        
        return image_preprocessed
    
    def predict(self, image):
        """
        Predict ISL sign from image
        With prediction smoothing as shown in notebook
        """
        if self.model is None:
            logger.error("Model not loaded")
            return None
        
        try:
            # Preprocess
            processed = self.preprocess_image(image)
            
            # Predict
            predictions = self.model.predict(processed, verbose=0)[0]
            
            # Get top prediction
            class_index = np.argmax(predictions)
            confidence = float(predictions[class_index])
            
            # Add to smoothing buffer
            self.prediction_buffer.append(class_index)
            
            # Get most common prediction from buffer
            final_index = max(set(self.prediction_buffer), key=self.prediction_buffer.count)
            predicted_label = self.class_names[final_index]
            
            # Confidence filter (important for 178 classes as per notebook)
            if confidence < self.confidence_threshold:
                predicted_label = "Detecting..."
            
            return {
                'sign': predicted_label,
                'confidence': confidence,
                'class_index': int(class_index),
                'is_valid': confidence >= self.confidence_threshold and predicted_label != "Detecting..."
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
    
    def predict_from_hand_crop(self, frame, hand_landmarks, padding=20):
        """
        Extract hand region from frame using landmarks and predict
        """
        try:
            h, w, _ = frame.shape
            
            # Get bounding box from landmarks
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            
            # Add padding
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            
            # Crop hand region
            hand_crop = frame[y_min:y_max, x_min:x_max]
            
            if hand_crop.size == 0:
                return None
            
            # Predict
            result = self.predict(hand_crop)
            if result:
                result['bbox'] = (x_min, y_min, x_max, y_max)
            
            return result
            
        except Exception as e:
            logger.error(f"Hand crop prediction error: {e}")
            return None
    
    def get_model_info(self):
        """Get model information"""
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "input_shape": self.model.input_shape,
            "output_shape": self.model.output_shape,
            "num_classes": len(self.class_names),
            "confidence_threshold": self.confidence_threshold,
            "model_path": self.model_path
        }

# Global instance
_isl_image_model = None

def get_isl_image_model():
    """Get or create global ISL image model instance"""
    global _isl_image_model
    if _isl_image_model is None:
        _isl_image_model = ISLImageModel()
    return _isl_image_model

def load_isl_image_model():
    """Load the ISL image model"""
    model = get_isl_image_model()
    return model.load_model()

def predict_sign(image):
    """Quick predict function"""
    model = get_isl_image_model()
    if model.model is None:
        if not model.load_model():
            return None
    return model.predict(image)

def predict_from_hand(frame, hand_landmarks):
    """Predict from hand landmarks"""
    model = get_isl_image_model()
    if model.model is None:
        if not model.load_model():
            return None
    return model.predict_from_hand_crop(frame, hand_landmarks)
