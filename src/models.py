import torch, torch.nn as nn
from transformers import AutoModel

class TextEncoder(nn.Module):
    def __init__(self, model_name): super().__init__(); self.model = AutoModel.from_pretrained(model_name)
    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]

class ImageEncoder(nn.Module):
    def __init__(self, model_name): super().__init__(); self.model = AutoModel.from_pretrained(model_name)
    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values)
        return out.last_hidden_state[:, 0, :]

class TextClassifier(nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.encoder = TextEncoder(model_name)
        hidden = self.encoder.model.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
    def forward(self, input_ids, attention_mask):
        x = self.encoder(input_ids, attention_mask)
        return self.head(x).squeeze(-1)

class ImageClassifier(nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.encoder = ImageEncoder(model_name)
        hidden = self.encoder.model.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
    def forward(self, pixel_values):
        x = self.encoder(pixel_values)
        return self.head(x).squeeze(-1)

class FusionHead(nn.Module):
    def __init__(self, in_dim, hidden=512, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

class FusionModel(nn.Module):
    def __init__(self, text_model_name, image_model_name, freeze_encoders=False):
        super().__init__()
        self.text = TextEncoder(text_model_name)
        self.image = ImageEncoder(image_model_name)
        if freeze_encoders:
            for p in self.text.parameters():  p.requires_grad = False
            for p in self.image.parameters(): p.requires_grad = False
        t_h = self.text.model.config.hidden_size
        i_h = self.image.model.config.hidden_size
        self.head = FusionHead(t_h + i_h)
    def forward(self, input_ids, attention_mask, pixel_values):
        t = self.text(input_ids, attention_mask)
        i = self.image(pixel_values)
        return self.head(torch.cat([t,i], dim=-1))
