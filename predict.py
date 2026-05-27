import tensorflow as tf
import numpy as np
import sys

IMG_SIZE = 256

CLASS_NAMES = [
    'apple_level_0',
    'apple_level_1',
    'apple_level_2',
    'potato_level_0',
    'potato_level_1',
    'potato_level_2'
]

# Cargar modelo
model = tf.keras.models.load_model("model/3")

# Obtener ruta de imagen
img_path = sys.argv[1]

# Leer imagen
img = tf.io.read_file(img_path)

# Decodificar imagen
img = tf.image.decode_image(
    img,
    channels=3,
    expand_animations=False
)

# Redimensionar
img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])

# NOTA: el modelo ya tiene Rescaling(1./255) incluido, NO normalizar manualmente

# Agregar dimensión batch
img = tf.expand_dims(img, axis=0)

# Predicción
preds = model.predict(img, verbose=0)

pred_idx = np.argmax(preds[0])
confidence = np.max(preds[0])

# Resultados
print(f'Predicción: {CLASS_NAMES[pred_idx]}')
print(f'Confianza: {confidence:.4f}')

print("\nProbabilidades:")
for cls, prob in zip(CLASS_NAMES, preds[0]):
    print(f'{cls}: {prob:.4f}')