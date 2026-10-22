import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
import pickle

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    "real_data/train", image_size=IMG_SIZE,
    batch_size=BATCH_SIZE, label_mode="binary")

val_ds = tf.keras.utils.image_dataset_from_directory(
    "real_data/val", image_size=IMG_SIZE,
    batch_size=BATCH_SIZE, label_mode="binary")

print("Classes:", train_ds.class_names)  # ['cancer','non_cancer'] → cancer=0

# Augmentation
data_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.15),
])

# Base model
base = EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False

inputs = layers.Input(shape=(224, 224, 3))
x = data_aug(inputs)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy",
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")]
)

# ⚠️ CORRECTED class weights — CANCER (class 0) la jada weight
class_weight = {0: 1.5, 1: 1.0}

AUTOTUNE = tf.data.AUTOTUNE
train_ds_p = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds_p = val_ds.cache().prefetch(AUTOTUNE)

print("\n🚀 Phase 1: Head training (base frozen)")
h1 = model.fit(train_ds_p, validation_data=val_ds_p,
               epochs=10, class_weight=class_weight)

print("\n🔧 Phase 2: Fine-tuning")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy",
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")]
)

h2 = model.fit(train_ds_p, validation_data=val_ds_p,
               epochs=10, class_weight=class_weight)

model.save("oral_cancer_model_v3.keras")
with open("history_v3.pkl","wb") as f:
    pickle.dump({"phase1":h1.history, "phase2":h2.history}, f)
print("\n✅ Model: oral_cancer_model_v3.keras")