# Configuração para execução no Google Colab
%matplotlib inline

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# Download do dataset Cats vs Dogs
dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
path_to_zip = tf.keras.utils.get_file("cats_and_dogs_filtered.zip", origin=dataset_url, extract=True)
dataset_path = os.path.join(os.path.dirname(path_to_zip), "cats_and_dogs_filtered")

# Diretórios do dataset
train_dir = os.path.join(dataset_path, "train")
validation_dir = os.path.join(dataset_path, "validation")

# Diretórios de gatos e cachorros no conjunto de treinamento
train_cats_dir = os.path.join(train_dir, "cats")
train_dogs_dir = os.path.join(train_dir, "dogs")

# Carregando o modelo VGG16 pré-treinado sem a camada de classificação
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(150, 150, 3))

# Congelando as camadas do modelo base
for layer in base_model.layers:
    layer.trainable = False

# Construindo o modelo de Transfer Learning
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")  # Saída binária para gatos e cachorros
])

# Compilando o modelo
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Preparação dos dados de treinamento e validação
train_datagen = image.ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

validation_datagen = image.ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary"
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary"
)

# Treinamento do modelo
history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=50
)

# Visualizando o desempenho
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(10)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Acurácia de Treinamento")
plt.plot(epochs_range, val_acc, label="Acurácia de Validação")
plt.legend(loc="lower right")
plt.title("Acurácia")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Loss de Treinamento")
plt.plot(epochs_range, val_loss, label="Loss de Validação")
plt.legend(loc="upper right")
plt.title("Loss")
plt.show()