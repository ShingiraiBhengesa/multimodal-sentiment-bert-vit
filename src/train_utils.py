import torch, torch.nn as nn, numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
criterion = nn.BCEWithLogitsLoss()

def _to_preds_labels(logits, labels):
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    preds = (probs >= 0.5).astype(int).tolist()
    labs  = labels.detach().cpu().numpy().astype(int).tolist()
    return preds, labs

def run_epoch_text(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()
    losses, P, L = [], [], []
    for b in tqdm(loader):
        ids, mask, y = b["input_ids"].to(device), b["attention_mask"].to(device), b["labels"].to(device)
        with torch.set_grad_enabled(train):
            logits = model(ids, mask); loss = criterion(logits, y)
        if train: optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(loss.item()); p,l = _to_preds_labels(logits, y); P+=p; L+=l
    return (sum(losses)/len(losses), accuracy_score(L,P), f1_score(L,P))

def run_epoch_image(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()
    losses, P, L = [], [], []
    for b in tqdm(loader):
        px, y = b["pixel_values"].to(device), b["labels"].to(device)
        with torch.set_grad_enabled(train):
            logits = model(px); loss = criterion(logits, y)
        if train: optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(loss.item()); p,l = _to_preds_labels(logits, y); P+=p; L+=l
    return (sum(losses)/len(losses), accuracy_score(L,P), f1_score(L,P))

def run_epoch_fusion(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()
    losses, P, L = [], [], []
    for b in tqdm(loader):
        ids, mask, px, y = b["input_ids"].to(device), b["attention_mask"].to(device), b["pixel_values"].to(device), b["labels"].to(device)
        with torch.set_grad_enabled(train):
            logits = model(ids, mask, px); loss = criterion(logits, y)
        if train: optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(loss.item()); p,l = _to_preds_labels(logits, y); P+=p; L+=l
    return (sum(losses)/len(losses), accuracy_score(L,P), f1_score(L,P))
