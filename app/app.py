import torch, gradio as gr
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor, AutoModel
from huggingface_hub import hf_hub_download
import os, torch.nn as nn

TEXT_REPO  = os.getenv("TEXT_REPO",  "<your-hf-username>/mm-sentiment-text-bert")
IMAGE_REPO = os.getenv("IMAGE_REPO", "<your-hf-username>/mm-sentiment-image-vit")
FUSION_REPO= os.getenv("FUSION_REPO","<your-hf-username>/mm-sentiment-fusion-head")
device = "cuda" if torch.cuda.is_available() else "cpu"

text_tok  = AutoTokenizer.from_pretrained(TEXT_REPO, use_fast=True)
img_proc  = AutoImageProcessor.from_pretrained(IMAGE_REPO)
text_enc  = AutoModel.from_pretrained(TEXT_REPO).to(device).eval()
img_enc   = AutoModel.from_pretrained(IMAGE_REPO).to(device).eval()
t_h = text_enc.config.hidden_size; i_h = img_enc.config.hidden_size

class FusionHead(nn.Module):
    def __init__(self, in_dim, hidden=512):
        super().__init__(); self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden,1))
    def forward(self, x): return self.net(x).squeeze(-1)

fusion_pth = hf_hub_download(repo_id=FUSION_REPO, repo_type="model", filename="fusion_head.pth")
fusion_head = FusionHead(t_h+i_h).to(device)
fusion_head.load_state_dict(torch.load(fusion_pth, map_location=device)); fusion_head.eval()

def predict(text, image):
    if not text or image is None:
        return {"Negative": 0.5, "Positive": 0.5}, 0.0
    enc = text_tok(text, padding="max_length", truncation=True, max_length=160, return_tensors="pt")
    for k in enc: enc[k] = enc[k].to(device)
    with torch.no_grad(): t_out = text_enc(**enc).last_hidden_state[:,0,:]
    if isinstance(image, str): image = Image.open(image).convert("RGB")
    else: image = image.convert("RGB")
    img = img_proc(images=image, return_tensors="pt")["pixel_values"].to(device)
    with torch.no_grad(): i_out = img_enc(pixel_values=img).last_hidden_state[:,0,:]
    with torch.no_grad():
        prob = torch.sigmoid(fusion_head(torch.cat([t_out,i_out], dim=-1))).item()
    return {"Negative": 1-prob, "Positive": prob}, prob

demo = gr.Interface(
    fn=predict,
    inputs=[gr.Textbox(lines=4, label="Review Text"), gr.Image(type="pil", label="Product Image")],
    outputs=[gr.Label(num_top_classes=2, label="Sentiment"), gr.Number(label="Positivity Probability")],
    title="Multimodal Sentiment (BERT + ViT)", description="Enter review text + product image."
)
if __name__ == "__main__": demo.launch()
