import os, torch, json
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from utils import (
    ensure_dir, get_device, load_json, save_json,
    save_confusion_matrix
)

DATA_DIR = "data"
RESULTS_DIR = "results"
MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.pth")
CLASSES_JSON = os.path.join(RESULTS_DIR, "classes.json")

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2

@torch.no_grad()
def main():
    device = get_device()
    classes = load_json(CLASSES_JSON)["classes"]

    test_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_tf)
    assert test_ds.classes == classes, "Las clases de test no coinciden con las de entrenamiento"

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # modelo
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()

    all_preds, all_true = [], []
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(1)
        all_preds.append(preds.cpu().numpy())
        all_true.append(y.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_preds)

    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    ensure_dir(RESULTS_DIR)
    save_json(report, os.path.join(RESULTS_DIR, "metrics_test.json"))

    save_confusion_matrix(y_true, y_pred, classes, os.path.join(RESULTS_DIR, "confusion_matrix.png"), normalize=True)
    print("Evaluación completada. Resultados en results/metrics_test.json y confusion_matrix.png")

if __name__ == "__main__":
    main()
