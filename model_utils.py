import os
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path

MODEL_PATH = "models/oral_cancer_model.h5"

def build_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    return model


def load_images_from_folder(folder_path, label):
    images = []
    labels = []

    for file in Path(folder_path).glob("*.*"):
        if file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        try:
            img = Image.open(file).convert("RGB").resize((224, 224))
            arr = np.array(img, dtype=np.float32) / 255.0
            images.append(arr)
            labels.append(label)
        except Exception:
            continue

    if len(images) == 0:
        raise ValueError(f"No valid images found in {folder_path}")

    return np.stack(images), np.array(labels, dtype=np.float32)


def train_model():
    train_non = Path("dataset/train/non_cancer")
    train_can = Path("dataset/train/cancer")
    val_non = Path("dataset/val/non_cancer")
    val_can = Path("dataset/val/cancer")

    if not train_non.exists() or not train_can.exists():
        raise FileNotFoundError("Train folders not found. Create dataset/train/non_cancer and dataset/train/cancer")

    if not val_non.exists() or not val_can.exists():
        raise FileNotFoundError("Validation folders not found. Create dataset/val/non_cancer and dataset/val/cancer")

    x_train = []
    y_train = []
    x_val = []
    y_val = []

    imgs, labels = load_images_from_folder(train_non, 0)
    x_train.extend(imgs)
    y_train.extend(labels)

    imgs, labels = load_images_from_folder(train_can, 1)
    x_train.extend(imgs)
    y_train.extend(labels)

    imgs, labels = load_images_from_folder(val_non, 0)
    x_val.extend(imgs)
    y_val.extend(labels)

    imgs, labels = load_images_from_folder(val_can, 1)
    x_val.extend(imgs)
    y_val.extend(labels)

    x_train = np.stack(x_train)
    y_train = np.array(y_train, dtype=np.float32)

    x_val = np.stack(x_val)
    y_val = np.array(y_val, dtype=np.float32)

    model = build_model()
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=20,
        batch_size=16
    )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print("Model trained and saved to:", MODEL_PATH)


if __name__ == "__main__":
    train_model()