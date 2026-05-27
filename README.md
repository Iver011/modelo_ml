# Clasificación de Enfermedades en Plantas (Apple & Potato)

Proyecto de clasificación de imágenes de hojas de manzana y papa usando **TensorFlow/Keras** con redes neuronales convolucionales (CNN). Clasifica en 6 categorías según el tipo de cultivo y nivel de severidad de la enfermedad.

---

## Problema

Identificar automáticamente enfermedades en cultivos de manzana y papa a partir de imágenes de hojas, clasificándolas en niveles de severidad (0, 1, 2).

---

## Dataset

### Original (PlantVillage)
- `training/PlantVillage/` — 2,152 imágenes de papa con 3 clases:

| Clase | Imágenes |
|---|---|
| `Potato___Early_blight` | 1,000 |
| `Potato___Late_blight` | 1,000 |
| `Potato___healthy` | 152 |

### Balanceado (Aumentado)
- `dataset_balanceado/` — 6 clases combinando manzana + papa, con split train/val/test:

| Clase | Train | Val | Test |
|---|---|---|---|
| `apple_level_0` | 394 | 84 | 85 |
| `apple_level_1` | 966 | 207 | 208 |
| `apple_level_2` | 1,187 | 255 | 255 |
| `potato_level_0` | 735 | 157 | 157 |
| `potato_level_1` | 840 | 180 | 180 |
| `potato_level_2` | 979 | 210 | 210 |
| **Total** | **5,101** | **1,093** | **1,095** |

---

## Arquitectura del Modelo

CNN secuencial con 6 bloques Conv2D + MaxPooling2D:

| Capa | Detalle |
|---|---|
| `Resizing` | Redimensiona a 256×256 |
| `Rescaling` | Normaliza píxeles a [0,1] |
| `RandomFlip` + `RandomRotation` | Data augmentation |
| `Conv2D(32, 3) + MaxPooling2D` | Bloque 1 |
| `Conv2D(64, 3) + MaxPooling2D` | Bloques 2-6 |
| `Flatten` | Aplanado |
| `Dropout(0.5)` | Regularización |
| `Dense(64, relu)` | Capa oculta |
| `Dropout(0.3)` | Regularización |
| `Dense(6, softmax)` | Salida (6 clases) |

- Optimizador: Adam
- Loss: `sparse_categorical_crossentropy`
- Métrica: accuracy
- Epochs: 25
- Batch size: 32

---

## Estructura del Proyecto

```
proyecto_ml/
├── training.ipynb              # Notebook inicial (solo papa, PlantVillage)
├── mode_v1.ipynb               # Notebook v1 (manzana + papa, 6 clases)
├── train_balanced.py           # Script de entrenamiento (reproducible)
├── predict.py                  # Script de inferencia
├── fix_notebook.py             # Utilidad para corregir celdas del notebook
├── manzana.jpg                 # Imagen de ejemplo para pruebas
├── training/
│   └── PlantVillage/           # Dataset original (papa, 3 clases)
├── dataset_balanceado/
│   ├── train/                  # Train set (5,101 imágenes, 6 clases)
│   ├── val/                    # Validation set (1,093 imágenes)
│   └── test/                   # Test set (1,095 imágenes)
├── model/
│   ├── 1/                      # Modelo versión 1
│   ├── 2/                      # Modelo versión 2
│   └── 3/                      # Modelo versión 3 (final)
├── venv/                       # Entorno virtual Python
└── README.md                   # Este archivo
```

---

## Instalación

1. Crear y activar entorno virtual:
```bash
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
```

2. Instalar dependencias:
```bash
pip install tensorflow numpy
```

---

## Uso

### Entrenar modelo
```bash
python train_balanced.py
```
Guarda el modelo en `model_balanced/`.

### Predecir con una imagen
```bash
python predict.py ruta/de/imagen.jpg
```
Ejemplo:
```bash
python predict.py manzana.jpg
```

Salida esperada:
```
Predicción: apple_level_0
Confianza: 0.9876

Probabilidades:
apple_level_0: 0.9876
apple_level_1: 0.0083
apple_level_2: 0.0021
potato_level_0: 0.0009
potato_level_1: 0.0005
potato_level_2: 0.0006
```

### Notebooks (Jupyter)
Abrir `mode_v1.ipynb` o `training.ipynb` en Jupyter Notebook/Lab.

---

## Resultados

| Métrica | Valor |
|---|---|
| Test Accuracy | ~85-90% |
| Clases | 6 |
| Imágenes totales | 7,289 |

---

## Clases

| Clase | Significado |
|---|---|
| `apple_level_0` | Manzana — sana o nivel bajo |
| `apple_level_1` | Manzana — nivel medio |
| `apple_level_2` | Manzana — nivel severo |
| `potato_level_0` | Papa — sana o nivel bajo |
| `potato_level_1` | Papa — nivel medio |
| `potato_level_2` | Papa — nivel severo |

---

## Modelos Guardados

En `model/` se encuentran 3 versiones exportadas en formato TensorFlow SavedModel:

| Versión | Descripción |
|---|---|
| `model/1` | Primer entrenamiento |
| `model/2` | Segunda iteración |
| `model/3` | Modelo final (usado por `predict.py`) |
