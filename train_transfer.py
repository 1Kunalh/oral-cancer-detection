import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import pickle

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ---------- Dataset ----------
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

print("Classes:", train_ds.class_names)

# ---------- Augmentation (overfitting kami karnyasathi) ----------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# ---------- Base Model (pre-trained) ----------
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # Aadhi freeze kara

# ---------- Full Model ----------
inputs = layers.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = layers.Rescaling(1./127.5, offset=-1)(x)   # MobileNetV2 sathi [-1, 1]
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)         # Flatten peksha chaan
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy",
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")]
)

model.summary()

# ---------- Class weights (non_cancer recall improve karnyasathi) ----------
# cancer=400, non_cancer=360 → thoda balance
class_weight = {0: 1.0, 1: 1.1}   # non_cancer la thoda jada weight

# ---------- Phase 1: Head training ----------
print("\n🚀 Phase 1: Training head (base frozen)...")
AUTOTUNE = tf.data.AUTOTUNE
train_ds_p = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds_p   = val_ds.cache().prefetch(AUTOTUNE)

history1 = model.fit(
    train_ds_p,
    validation_data=val_ds_p,
    epochs=10,
    class_weight=class_weight
)

# ---------- Phase 2: Fine-tuning (top layers unfreeze) ----------
print("\n🔧 Phase 2: Fine-tuning top layers...")
base_model.trainable = True
# Fakt shevtache 30 layers unfreeze kara
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy",
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")]
)

history2 = model.fit(
    train_ds_p,
    validation_data=val_ds_p,
    epochs=10,
    class_weight=class_weight
)

# ---------- Save ----------
model.save("oral_cancer_model_v2.keras")
print("\n✅ Model saved as oral_cancer_model_v2.keras")

# History save kara (graphs sathi)
with open("history_v2.pkl", "wb") as f:
    pickle.dump({"phase1": history1.history, "phase2": history2.history}, f)
print("✅ History saved as history_v2.pkl")