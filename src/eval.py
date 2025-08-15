import torch, seaborn as sns, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from src.models import FusionModel
from src import config
from src.data_utils import MultimodalDataset, collate_fn

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_ds = MultimodalDataset(f"{config.DATA_DIR}/test.csv", config.BERT_MODEL, config.VIT_MODEL, max_length=config.MAX_LEN)
    loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    model = FusionModel(config.TEXT_CKPT, config.IMAGE_CKPT, freeze_encoders=True).to(device)
    model.head.load_state_dict(torch.load(config.FUSION_CKPT, map_location=device))
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for b in loader:
            ids, mask, px = b["input_ids"].to(device), b["attention_mask"].to(device), b["pixel_values"].to(device)
            y = b["labels"].to(device)
            logits = model(ids, mask, px); p = (torch.sigmoid(logits)>=0.5).int().cpu().tolist()
            preds += p; labels += y.int().cpu().tolist()
    acc = accuracy_score(labels, preds); f1 = f1_score(labels, preds)
    print("Accuracy:", acc, "F1:", f1)
    print(classification_report(labels, preds, digits=3))
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=["neg","pos"], yticklabels=["neg","pos"], cbar=False)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.tight_layout(); plt.savefig("confusion_matrix.png"); print("Saved confusion_matrix.png")
if __name__ == "__main__": main()
