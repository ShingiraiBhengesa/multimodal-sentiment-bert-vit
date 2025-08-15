import os, torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from src.data_utils import MultimodalDataset, collate_fn
from src.models import FusionModel
from src import config
from src.train_utils import run_epoch_fusion

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert os.path.isdir(config.TEXT_CKPT) and os.path.isdir(config.IMAGE_CKPT), "Run train_text.py and train_image.py first."
    train_ds = MultimodalDataset(f"{config.DATA_DIR}/train.csv", config.BERT_MODEL, config.VIT_MODEL, max_length=config.MAX_LEN)
    val_ds   = MultimodalDataset(f"{config.DATA_DIR}/val.csv",   config.BERT_MODEL, config.VIT_MODEL, max_length=config.MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    model = FusionModel(config.TEXT_CKPT, config.IMAGE_CKPT, freeze_encoders=True).to(device)
    opt = AdamW(model.head.parameters(), lr=config.LR_FUSION)
    best = 0.0
    os.makedirs("checkpoints", exist_ok=True)
    for e in range(config.EPOCHS_FUSION):
        tr = run_epoch_fusion(model, train_loader, opt, device, True)
        va = run_epoch_fusion(model, val_loader,   opt, device, False)
        print(f"Epoch {e+1}: train loss {tr[0]:.4f}, acc {tr[1]:.3f}, f1 {tr[2]:.3f} | val loss {va[0]:.4f}, acc {va[1]:.3f}, f1 {va[2]:.3f}")
        if va[2] > best:
            best = va[2]; torch.save(model.head.state_dict(), config.FUSION_CKPT); print('Saved ->', config.FUSION_CKPT)
if __name__ == "__main__": main()
