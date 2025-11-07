import os, torch
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import (
    ensure_dir, get_device, load_json, denormalize_img
)

DATA_DIR = "data"
RESULTS_DIR = "results"
MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.pth")
CLASSES_JSON = os.path.join(RESULTS_DIR, "classes.json")
OUT_DIR = os.path.join(RESULTS_DIR, "gradcam")

IMG_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 2
SAMPLES_PER_CLASS = 2  # imágenes por clase intentar guardar

class GradCAM:
    """Grad-CAM simple sobre la última capa conv (layer4) de ResNet50."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(fwd_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(bwd_hook))

    def __call__(self, scores, class_idx):
        # scores: tensor shape (N, num_classes)
        self.model.zero_grad(set_to_none=True)
        loss = scores[:, class_idx].sum()
        loss.backward(retain_graph=True)  # grads on last conv

        grads = self.gradients           # (N, C, H, W)
        acts = self.activations          # (N, C, H, W)
        weights = grads.mean(dim=(2,3), keepdim=True)  # GAP sobre H,W
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (N,1,H,W)
        cam = torch.relu(cam)
        # normalizar por imagen
        N, _, H, W = cam.shape
        cam = cam.view(N, -1)
        cam = (cam - cam.min(dim=1, keepdim=True).values) / (cam.max(dim=1, keepdim=True).values - cam.min(dim=1, keepdim=True).values + 1e-8)
        cam = cam.view(N, 1, H, W)
        return cam  # [0,1]

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()

@torch.no_grad()
def main():
    device = get_device()
    classes = load_json(CLASSES_JSON)["classes"]

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    # usar VALIDATION para ejemplos
    ds = datasets.ImageFolder(os.path.join(DATA_DIR, "validation"), transform=tf)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    # Modelo
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()

    # target layer
    target_layer = model.layer4[-1].conv3
    cam = GradCAM(model, target_layer)

    ensure_dir(OUT_DIR)
    saved_per_class = {c:0 for c in classes}

    for x, y in loader:
        x = x.to(device)
        scores = model(x)  # (B, num_classes)
        preds = scores.argmax(1)

        # generar CAM para la clase predicha
        cams = cam(scores, preds)  # (B,1,H,W) en [0,1]

        for i in range(x.size(0)):
            cls_name = classes[int(preds[i].item())]
            if saved_per_class[cls_name] >= SAMPLES_PER_CLASS:
                continue

            img = denormalize_img(x[i])
            heat = cams[i,0].detach().cpu().numpy()
            heat = np.uint8(255 * heat)
            heat = plt.cm.jet(heat)[:,:,:3]  # RGB

            overlay = (0.6*img + 0.4*heat)
            overlay = np.clip(overlay, 0, 1)

            out_path = os.path.join(OUT_DIR, f"{cls_name}_{saved_per_class[cls_name]+1}.png")
            plt.imsave(out_path, overlay)
            saved_per_class[cls_name] += 1

        if all(saved_per_class[c] >= SAMPLES_PER_CLASS for c in classes):
            break

    cam.remove_hooks()
    print(f"Grad-CAM listo. Imágenes en: {OUT_DIR}")

if __name__ == "__main__":
    main()
