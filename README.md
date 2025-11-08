# Clasificacion-Vehiculos-MLT

## Descripción del Proyecto
Este proyecto implementa un sistema de clasificación automática de vehículos militares utilizando Transfer Learning, específicamente con una ResNet-50 preentrenada en ImageNet, seguida de un proceso de fine-tuning.

El objetivo es identificar correctamente la clase de cada vehículo militar (tanques, APC, artillería autopropulsada, camiones tácticos, LAVs, etc.) basándonos en el dataset Military Vehicles de Kaggle.

El entrenamiento, evaluación, visualización e interpretabilidad del modelo se encuentran completamente automatizados mediante los scripts en `src/`.

## Estructura del Proyecto

```
Clasificacion-Vehiculos-MLT/
├── data/                    # Dataset organizado en train/validation/test/
│   ├── train/
│   ├── validation/
│   └── test/
├── src/
│   ├── train.py             # Entrenamiento y fine-tuning
│   ├── eval.py              # Evaluación final + matriz de confusión
│   ├── gradcam.py           # Interpretabilidad (Grad-CAM)
│   ├── utils.py             # Funciones auxiliares
├── experiments/             # Configuraciones de entrenamiento
├── results/                 # Checkpoints, curvas, métricas y visualizaciones
│   ├── best_model.pth
│   ├── curves.png
│   ├── confusion_matrix.png
│   ├── gradcam/
│   ├── metrics_val_and_quicktest.json
│   └── metrics_test.json
├── requirements.txt
├── .gitignore
└── README.md
```

## Instalación

1. Clona el repositorio:
   ```
   git clone <url del repo>
   cd Clasificacion-Vehiculos-MLT
   ```

2. Crear y activar un entorno virtual:
   ```
   python -m venv .venv

   # Para Linux/macOS
   source .venv/bin/activate

   # Para Windows (PowerShell)
   .venv\Scripts\activate
   ```

3. Instalar dependencias:
   ```
   pip install -r requirements.txt
   ```
   Verificar GPU
   ```
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name())"
   ```
   Si ves tu GPU (ej. RTX 4060), estás listo.

## Preparación del Dataset
Este proyecto utiliza datos del dataset:

🔗[Military Vehicles Dataset](https://www.kaggle.com/datasets/amanrajbose/millitary-vechiles?select=dataset)

Organiza el dataset en la carpeta data de esta forma:
```
data/
│
├── train/
├── validation/
└── test/
```
Cada carpeta debe tener subcarpetas por clase:
```
train/
  ├── tanks/
  ├── anti-aircraft/
  ├── armored personnel carriers/
  ├── light utility vehicles/
  └── etc...
```

## Entrenamiento del Modelo
Para entrenar desde cero:
```
python src/train.py
```
Esto ejecuta:
- Transfer Learning (congelando ResNet-50)
- Fine-tuning de la capa layer4
- Early stopping
- Guardado del mejor modelo
- Generación automática de:
  - `results/best_model.pth`
  - `results/curves.png`
  - `results/metrics_val_and_quicktest.json`
  - `experiments/*.json`

## Evaluación del Modelo
```
python src/eval.py
```
Esto genera:
-  Matriz de confusión → `results/confusion_matrix.png`
-  Métricas finales → `results/metrics_test.json`

Las métricas alcanzadas:
- Accuracy validation ≈ 96.55%
- F1-macro validation ≈ 96.40%
- Accuracy test ≈ 95.97%
- F1-macro test ≈ 95.81%

## Interpretabilidad con Grad-CAM
Para visualizar qué partes de la imagen usa el modelo:
```
python src/gradcam.py
```
Se generarán mapas Grad-CAM en:
`results/gradcam/`

Incluye:
- Vehículos con heatmaps superpuestos
- Casos bien clasificados
- Casos difíciles
- Análisis visual de torretas, ruedas, cañones, cabinas, etc.

## Resultados Principales
### Curvas de entrenamiento
Ubicadas en:
`results/curves.png`
- Train loss disminuye de forma estable.
- Validation loss se estabiliza (~0.12).
- No hay sobreajuste visible.

## Matriz de confusión
Ubicación:
`results/confusion_matrix.png`
- El modelo domina casi todas las clases.
- Confusiones esperadas entre vehículos similares (APC vs IFV).

## Interpretabilidad (Grad-CAM)
Ubicación:
`results/gradcam/`
- El modelo se centra en rasgos relevantes: torretas, ángulos frontales, cabinas, orugas.

