import os, torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from src.data_utils import MultimodalDataset, collate_fn
from src.models import TextClassifier
from src import config
from src.train_utils import run_epoch_text

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(config.TEXT_CKPT, exist_ok=True)
    train_ds = MultimodalDataset(f"{config.DATA_DIR}/train.csv", config.BERT_MODEL, config.VIT_MODEL, max_length=config.MAX_LEN)
    val_ds   = MultimodalDataset(f"{config.DATA_DIR}/val.csv",   config.BERT_MODEL, config.VIT_MODEL, max_length=config.MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    model = TextClassifier(config.BERT_MODEL).to(device)
    opt = AdamW(model.parameters(), lr=config.LR_TEXT)
    best = 0.0
    for e in range(config.EPOCHS_SINGLE):
        tr = run_epoch_text(model, train_loader, opt, device, True)
        va = run_epoch_text(model, val_loader,   opt, device, False)
        print(f"Epoch {e+1}: train loss {tr[0]:.4f}, acc {tr[1]:.3f}, f1 {tr[2]:.3f} | val loss {va[0]:.4f}, acc {va[1]:.3f}, f1 {va[2]:.3f}")
        if va[2] > best:
            best = va[2]; model.encoder.model.save_pretrained(config.TEXT_CKPT); print('Saved ->', config.TEXT_CKPT)
if __name__ == "__main__": main()
