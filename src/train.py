import os, time, copy, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report

from utils import (
    ensure_dir, set_seed, get_device, save_json, get_class_names,
    plot_training_curves, compute_metrics
)

# Configuración
DATA_DIR = "data"  # contiene train/ validation/ test/
EXPER_DIR = "experiments"
RESULTS_DIR = "results"
MODEL_OUT = os.path.join(RESULTS_DIR, "best_model.pth")
CLASSES_JSON = os.path.join(RESULTS_DIR, "classes.json")
HPARAMS_YAML = os.path.join(EXPER_DIR, "resnet50_headonly.json")  # json para simplicidad
SEED = 42

BATCH_SIZE = 32
NUM_WORKERS = 2
EPOCHS = 12
HEAD_ONLY_EPOCHS = 4      # epochs con encoder congelado
LR_HEAD = 1e-3
LR_UNFREEZE = 3e-5
PATIENCE = 4              # early stopping

IMG_SIZE = 224

def build_dataloaders():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.2,0.2,0.2,0.05)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "validation"), transform=val_tf)
    test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=val_tf)

    class_names = get_class_names(train_ds)
    assert class_names == val_ds.classes == test_ds.classes, "Las clases deben coincidir en train/val/test"

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, val_loader, test_loader, class_names

def make_model(num_classes: int):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # congelar todo
    for p in model.parameters():
        p.requires_grad = False
    # reemplazar head
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, num_classes)
    return model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    all_preds, all_true = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += x.size(0)
        all_preds.append(preds.cpu().numpy())
        all_true.append(y.cpu().numpy())
    loss = loss_sum/total
    acc = correct/total
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_preds)
    f1m = compute_metrics(y_true, y_pred)["macro_f1"]
    return loss, acc, f1m, y_true, y_pred

def main():
    set_seed(SEED)
    ensure_dir(EXPER_DIR)
    ensure_dir(RESULTS_DIR)

    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, class_names = build_dataloaders()
    save_json({"classes": class_names}, CLASSES_JSON)

    model = make_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()

    # HEAD-ONLY
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=LR_HEAD)
    history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
    best_val_f1 = -1
    best_weights = None
    epochs_no_improve = 0

    print("\n==> Entrenando HEAD-ONLY...")
    for epoch in range(1, HEAD_ONLY_EPOCHS+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        print(f"[{epoch:02d}/{HEAD_ONLY_EPOCHS}] loss {tr_loss:.4f}/{val_loss:.4f} | acc {tr_acc:.3f}/{val_acc:.3f} | f1 {val_f1:.3f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save(best_weights, MODEL_OUT)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("Early stopping (head-only)")
                break

    # UNFREEZE parcial (ultimos bloques)
    print("\n==> Fine-tuning: unfreeze parcial (layer4)...")
    for name, p in model.named_parameters():
        if name.startswith("layer4") or name.startswith("fc"):
            p.requires_grad = True

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_UNFREEZE)

    total_epochs = EPOCHS
    for epoch in range(HEAD_ONLY_EPOCHS+1, total_epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        print(f"[{epoch:02d}/{total_epochs}] loss {tr_loss:.4f}/{val_loss:.4f} | acc {tr_acc:.3f}/{val_acc:.3f} | f1 {val_f1:.3f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save(best_weights, MODEL_OUT)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("Early stopping (fine-tuning)")
                break

    # Cargar mejor modelo y evaluar en VALIDATION & TEST
    model.load_state_dict(torch.load(MODEL_OUT, map_location=device))
    val_loss, val_acc, val_f1, yv, pv = evaluate(model, val_loader, criterion, device)
    test_loss, test_acc, test_f1, yt, pt = evaluate(model, test_loader, criterion, device)

    # Guardar historia y metricas
    plot_training_curves(history, os.path.join(RESULTS_DIR, "curves.png"))
    save_json({
        "config": {
            "batch_size": BATCH_SIZE, "epochs": EPOCHS,
            "head_only_epochs": HEAD_ONLY_EPOCHS,
            "lr_head": LR_HEAD, "lr_unfreeze": LR_UNFREEZE,
            "img_size": IMG_SIZE, "seed": SEED
        },
        "val": {"loss": float(val_loss), "accuracy": float(val_acc), "macro_f1": float(val_f1)},
        "test_quick": {"loss": float(test_loss), "accuracy": float(test_acc), "macro_f1": float(test_f1)},
        "best_model_path": MODEL_OUT
    }, os.path.join(RESULTS_DIR, "metrics_val_and_quicktest.json"))

    # Guardar hparams como json (para experiments/)
    save_json({
        "model": "resnet50",
        "freeze": "all_then_unfreeze_layer4",
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr_head": LR_HEAD,
        "lr_unfreeze": LR_UNFREEZE
    }, HPARAMS_YAML)

    print("\nEntrenamiento finalizado. Mejor checkpoint guardado en:", MODEL_OUT)

if __name__ == "__main__":
    main()
