import torch, pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoImageProcessor
from PIL import Image

class MultimodalDataset(Dataset):
    def __init__(self, csv_path, tokenizer_name, image_processor_name, max_length=160):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.image_proc = AutoImageProcessor.from_pretrained(image_processor_name)
        self.max_length = max_length

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["text"])
        img_path = row["image_path"]
        label = int(row["label"])

        enc = self.tokenizer(text, padding="max_length", truncation=True,
                             max_length=self.max_length, return_tensors="pt")
        image = Image.open(img_path).convert("RGB")
        img_enc = self.image_proc(images=image, return_tensors="pt")

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pixel_values":   img_enc["pixel_values"].squeeze(0),
            "labels":         torch.tensor(label, dtype=torch.float)
        }

def collate_fn(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        out[k] = torch.stack([x[k] for x in batch])
    return out
