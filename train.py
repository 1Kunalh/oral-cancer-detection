import tensorflow as tf
from tensorflow.keras import layers, models

# ---------- Dataset Load ----------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    "real_data/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "real_data/val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

print("Class names:", train_ds.class_names)
# Expected: ['cancer', 'non_cancer']
# class 0 = cancer, class 1 = non_cancer (alphabetical)

# ---------- Performance tuning ----------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ---------- Normalization layer ----------
normalization_layer = layers.Rescaling(1./255)

# ---------- Model ----------
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    normalization_layer,

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------- Train ----------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# ---------- Save model ----------
model.save("oral_cancer_model.keras")
print("Model saved as oral_cancer_model.keras")