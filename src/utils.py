import os, json, random
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import itertools

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_json(obj: Dict, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_class_names(datasets_dict) -> List[str]:
    return list(datasets_dict.classes)

def plot_training_curves(history: Dict[str, List[float]], out_path: str):
    ensure_dir(os.path.dirname(out_path))
    plt.figure(figsize=(8,5))
    epochs = range(1, len(history["train_loss"])+1)
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Época")
    plt.ylabel("Pérdida / Exactitud")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_f1": f1m}

def save_confusion_matrix(y_true, y_pred, class_names: List[str], out_path: str, normalize: bool = True):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Matriz de confusión" + (" (norm.)" if normalize else ""))
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=8)

    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=140)
    plt.close()

def denormalize_img(t: torch.Tensor) -> np.ndarray:
    """t: (C,H,W) tensor normalized with ImageNet stats -> returns HxWxC [0,1]"""
    mean = torch.tensor(IMAGENET_MEAN, device=t.device).view(3,1,1)
    std  = torch.tensor(IMAGENET_STD,  device=t.device).view(3,1,1)
    x = t * std + mean
    x = torch.clamp(x, 0, 1)
    x = x.permute(1,2,0).detach().cpu().numpy()
    return x
