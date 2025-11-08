import os, torch
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
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
            # out: (N, C, H, W)
            self.activations = out

        def bwd_hook(module, grad_in, grad_out):
            # grad_out[0]: (N, C, H, W)
            self.gradients = grad_out[0]

        self.hook_handles.append(self.target_layer.register_forward_hook(fwd_hook))
        # usar full_backward_hook (el backward_hook clásico está deprecado)
        self.hook_handles.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def __call__(self, scores, class_idx):
        # scores: (N, num_classes); class_idx: (N,)
        self.model.zero_grad(set_to_none=True)
        # Selecciona el logit de la clase predicha por elemento y suma para backprop
        loss = scores[torch.arange(scores.size(0)), class_idx].sum()
        loss.backward(retain_graph=True)

        grads = self.gradients          # (N, C, H, W)
        acts  = self.activations        # (N, C, H, W)
        # pesos por GAP en H,W
        weights = grads.mean(dim=(2,3), keepdim=True)            # (N, C, 1, 1)
        cam = (weights * acts).sum(dim=1, keepdim=True).relu()   # (N, 1, H, W)

        # normaliza por imagen a [0,1]
        N = cam.size(0)
        cam = cam.view(N, -1)
        cam = (cam - cam.min(dim=1, keepdim=True).values) / (
              cam.max(dim=1, keepdim=True).values - cam.min(dim=1, keepdim=True).values + 1e-8)
        cam = cam.view(N, 1, self.activations.shape[2], self.activations.shape[3])
        return cam

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()

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
        cams = F.interpolate(cams, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)  # (B,1,224,224)

        for i in range(x.size(0)):
            cls_name = classes[int(preds[i].item())]
            if saved_per_class[cls_name] >= SAMPLES_PER_CLASS:
                continue

            img = denormalize_img(x[i])
            heat = cams[i, 0].detach().cpu().numpy()
            heat_u8 = (heat * 255).astype(np.uint8)
            heat_rgb = plt.cm.jet(heat_u8)[..., :3]

            overlay = 0.6 * img + 0.4 * heat_rgb
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
