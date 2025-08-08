import torch
import torch.nn as nn
import torchvision.models as models
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SignLanguageModel(nn.Module):
    def __init__(self, num_classes: int = 39):
        super().__init__()
        self.num_classes = num_classes
        
        # CNN backbone
        self.cnn_backbone = models.efficientnet_v2_s(weights='DEFAULT')
        self.cnn_backbone.classifier = nn.Identity()
        
        # Positional embedding
        self.positional_embedding = nn.Embedding(16, 1280)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=1280, nhead=4, dim_feedforward=256, dropout=0.2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Process frames
        x = x.reshape(-1, 3, 64, 64)  # (bs*seq, c, h, w)
        x = self.cnn_backbone(x)  # (bs*seq, 1280)
        x = x.reshape(batch_size, seq_len, -1)  # (bs, seq, 1280)
        
        # Add positional embeddings
        positions = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)
        x = x + self.positional_embedding(positions)
        
        # Transformer
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        
        return self.classifier(x)

    @classmethod
    def from_pretrained(cls, model_path: str, device: str = "cpu") -> Dict[str, Any]:
        """Load pretrained model with configuration"""
        try:
            device = torch.device(device)
            checkpoint = torch.load(model_path, map_location=device)
            
            # Initialize model
            model = cls(num_classes=checkpoint['config']['num_classes'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval().to(device)
            
            logger.info(f"Loaded model from {model_path}")
            return {
                'model': model,
                'label_map': checkpoint.get('label_map', {}),
                'config': checkpoint.get('config', {})
            }
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise