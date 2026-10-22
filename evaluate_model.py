import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Model load
model = tf.keras.models.load_model("oral_cancer_model_v3.keras")

# Validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    "real_data/val",
    image_size=(224, 224),
    batch_size=32,
    label_mode="binary",
    shuffle=False
)

# Evaluate — v2 madhe 4 metrics ahet, so dict madhe ghe
results = model.evaluate(val_ds, return_dict=True)
print("\n===== Evaluation =====")
for k, v in results.items():
    print(f"{k}: {v:.4f}")
print("======================\n")

# Predictions
y_true = np.concatenate([y.numpy() for _, y in val_ds], axis=0).flatten()
y_prob = model.predict(val_ds).flatten()
THRESHOLD = 0.62   # Cancer recall priority
y_pred = (y_prob > THRESHOLD).astype(int)
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=["cancer", "non_cancer"]))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["cancer", "non_cancer"],
            yticklabels=["cancer", "non_cancer"])
plt.xlabel("Predicted"); plt.ylabel("Actual")
acc = results.get("accuracy", 0)
plt.title(f"Confusion Matrix (v2) — Acc: {acc*100:.2f}%")
plt.tight_layout()
plt.savefig("confusion_matrix_v2.png", dpi=150)
print("📊 Saved: confusion_matrix_v2.png")
plt.show()