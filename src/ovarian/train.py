"""Fold training for the CNN and ResNet18 (same recipe), and fold-ensemble inference."""
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from .config import TRAIN_PROTOCOL
from .data import UltrasoundDataset, build_transforms
from .models import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_proba(model, df, batch_size=32, device=DEVICE):
    _, tf = build_transforms("none")
    loader = DataLoader(UltrasoundDataset(df, tf), batch_size=batch_size, shuffle=False)
    model.eval().to(device)
    out = []
    for x, _ in loader:
        out.append(torch.softmax(model(x.to(device)).float(), 1).cpu().numpy())
    return np.concatenate(out)


def train_fold(model_name, cfg, dev_df, fold, out_dir, seed=42, max_epochs=None,
               patience=None, num_workers=0, device=DEVICE, pretrained=True):
    """Train on folds != `fold`, validate on `fold`. Saves best_model.pth, history.csv,
    summary.json in out_dir/fold{fold}. Skips the fold if summary.json already exists."""
    P = TRAIN_PROTOCOL
    max_epochs = max_epochs or P["max_epochs"]
    patience = patience or P["early_stopping_patience"]
    out = Path(out_dir) / f"fold{fold}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        return json.loads((out / "summary.json").read_text())

    set_seed(seed + fold)
    train_df, val_df = dev_df[dev_df.fold != fold], dev_df[dev_df.fold == fold]
    train_tf, val_tf = build_transforms(cfg["augmentation"])
    gen = torch.Generator().manual_seed(seed + fold)
    tr = DataLoader(UltrasoundDataset(train_df, train_tf), batch_size=cfg["batch_size"],
                    shuffle=True, drop_last=True, generator=gen, num_workers=num_workers)
    va = DataLoader(UltrasoundDataset(val_df, val_tf), batch_size=32, shuffle=False,
                    num_workers=num_workers)

    model = build_model(model_name, pretrained=pretrained, **cfg).to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                       patience=4, min_lr=1e-7)
    amp = device.type == "cuda" and P["mixed_precision"]
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    hist, best_f1, best_ep, bad, t0 = [], -1.0, 0, 0, time.time()
    for ep in range(1, max_epochs + 1):
        model.train(); loss_sum = n = correct = 0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                logits = model(x); loss = crit(logits, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            loss_sum += loss.item() * len(y); n += len(y)
            correct += (logits.argmax(1) == y).sum().item()
        prob = predict_proba(model, val_df, device=device)  # val_tf == eval transform
        y_true, y_pred = val_df["label"].values, prob.argmax(1)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        sched.step(f1)
        hist.append(dict(epoch=ep, train_loss=loss_sum / n, train_acc=correct / n,
                         val_acc=accuracy_score(y_true, y_pred), val_macro_f1=f1,
                         lr=opt.param_groups[0]["lr"]))
        if f1 > best_f1 + P["min_delta"]:
            best_f1, best_ep, bad = f1, ep, 0
            torch.save(model.state_dict(), out / "best_model.pth")
        else:
            bad += 1
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        print(f"[{model_name} fold{fold}] ep {ep:03d} train_acc {correct/n:.3f} "
              f"val_acc {hist[-1]['val_acc']:.3f} val_F1 {f1:.3f} best {best_f1:.3f}@{best_ep}")
        if bad >= patience:
            break

    summary = dict(model=model_name, fold=fold, best_epoch=best_ep, best_val_macro_f1=best_f1,
                   epochs_run=len(hist), minutes=round((time.time() - t0) / 60, 2), config=cfg)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def ensemble_predict(model_name, cfg, df, run_dir, device=DEVICE):
    """Average the softmax of every fold's best checkpoint -> (n, 5) probabilities."""
    probs = []
    for ck in sorted(Path(run_dir).glob("fold*/best_model.pth")):
        m = build_model(model_name, pretrained=False, **cfg)
        m.load_state_dict(torch.load(ck, map_location=device))
        probs.append(predict_proba(m, df, device=device))
    if not probs:
        raise FileNotFoundError(f"no fold checkpoints in {run_dir}")
    return np.mean(probs, axis=0)
