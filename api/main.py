
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import uvicorn
import numpy as np
from PIL import Image
import tensorflow as tf
from io import BytesIO

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo
MODEL = tf.keras.models.load_model("../model/3")

CLASS_NAMES = [
    'apple_level_0',
    'apple_level_1',
    'apple_level_2',
    'potato_level_0',
    'potato_level_1',
    'potato_level_2'
]

IMAGE_SIZE = 256


@app.get("/ping")
async def ping():
    return "Holaa, Estoy en vivo"


# Leer y redimensionar imagen
def read_file_as_image(data) -> np.ndarray:

    image = Image.open(BytesIO(data))

    # convertir a RGB por si suben PNG o imágenes en otro formato
    image = image.convert("RGB")

    # redimensionar
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    # convertir a array numpy
    image = np.array(image)

    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = read_file_as_image(await file.read())

    # agregar dimensión batch
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)