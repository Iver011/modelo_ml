import json

with open('mode_v1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb['cells'][3]['source'] = [
    'def build_model():\n',
    '    model = keras.Sequential([\n',
    '        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),\n',
    '        layers.Resizing(IMG_SIZE, IMG_SIZE),\n',
    '        layers.Rescaling(1./255),\n',
    '        data_augmentation,\n',
    '        layers.Conv2D(32, 3, activation="relu"),\n',
    '        layers.MaxPooling2D(),\n',
    '        layers.Conv2D(64, 3, activation="relu"),\n',
    '        layers.MaxPooling2D(),\n',
    '        layers.Conv2D(64, 3, activation="relu"),\n',
    '        layers.MaxPooling2D(),\n',
    '        layers.Conv2D(64, 3, activation="relu"),\n',
    '        layers.MaxPooling2D(),\n',
    '        layers.Conv2D(64, 3, activation="relu"),\n',
    '        layers.MaxPooling2D(),\n',
    '        layers.Conv2D(64, 3, activation="relu"),\n',
    '        layers.MaxPooling2D(),\n',
    '        layers.Flatten(),\n',
    '        layers.Dropout(0.5),\n',
    '        layers.Dense(64, activation="relu"),\n',
    '        layers.Dropout(0.3),\n',
    '        layers.Dense(NUM_CLASSES, activation="softmax"),\n',
    '    ])\n',
    '    return model\n',
]

nb['cells'][13]['source'] = [
    'model_version=3\n',
    'model.save(f"./model/{model_version}")\n',
]

with open('mode_v1.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Notebook updated OK')
