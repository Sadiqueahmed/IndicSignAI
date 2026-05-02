import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F  # Add at the top

class SignLanguageModel(nn.Module):
    def __init__(self, num_classes=39):  # Now matches your 39 classes
        super(SignLanguageModel, self).__init__()
        
        # CNN backbone (EfficientNetV2-S as closest to V2B0)
        self.cnn_backbone = models.efficientnet_v2_s(weights='DEFAULT')
        self.cnn_backbone.classifier = nn.Identity()  # Remove final classification layer
        
        # Positional embedding
        self.positional_embedding = nn.Embedding(16, 1280)  # EfficientNetV2-S output is 1280
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=1280, nhead=4, dim_feedforward=256, dropout=0.2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Classification head
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(1280, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Process each frame through CNN
        x = x.reshape(-1, 3, 64, 64)  # (bs*seq, c, h, w)
        x = self.cnn_backbone(x)  # (bs*seq, 1280)
        x = x.reshape(batch_size, seq_len, -1)  # (bs, seq, 1280)
        
        # Add positional embeddings
        positions = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)
        pos_emb = self.positional_embedding(positions)
        x = x + pos_emb
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (bs, 1280)
        
        # Classification head
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x