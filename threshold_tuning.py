import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

model = tf.keras.models.load_model("oral_cancer_model_v2.keras")

val_ds = tf.keras.utils.image_dataset_from_directory(
    "real_data/val", image_size=(224,224), batch_size=32,
    label_mode="binary", shuffle=False
)

y_true = np.concatenate([y.numpy() for _, y in val_ds], axis=0).flatten()
y_prob = model.predict(val_ds).flatten()

# ROC-AUC
auc = roc_auc_score(y_true, y_prob)
print(f"📈 ROC-AUC: {auc:.4f}\n")

# Different thresholds try kara
print("="*60)
print(f"{'Threshold':<12}{'Cancer Recall':<16}{'Non-Cancer Recall':<20}{'Accuracy':<10}")
print("="*60)

best_thresh = 0.5
best_score = 0

for t in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
    y_pred = (y_prob > t).astype(int)
    # y=0 cancer, y=1 non_cancer (alphabetical)
    cancer_recall = ((y_pred == 0) & (y_true == 0)).sum() / (y_true == 0).sum()
    noncancer_recall = ((y_pred == 1) & (y_true == 1)).sum() / (y_true == 1).sum()
    acc = (y_pred == y_true).mean()
    print(f"{t:<12.2f}{cancer_recall:<16.3f}{noncancer_recall:<20.3f}{acc:<10.3f}")
    
    # Balance point: both recalls >= 0.85
    if cancer_recall >= 0.85 and noncancer_recall >= 0.85:
        if (cancer_recall + noncancer_recall) > best_score:
            best_score = cancer_recall + noncancer_recall
            best_thresh = t

print("="*60)
print(f"\n✅ Best balanced threshold: {best_thresh}")
print("   (dono recalls >= 0.85)\n")

# Best threshold cha detailed report
y_pred_best = (y_prob > best_thresh).astype(int)
print(f"Classification Report @ threshold={best_thresh}:")
print(classification_report(y_true, y_pred_best, target_names=["cancer", "non_cancer"]))

cm = confusion_matrix(y_true, y_pred_best)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["cancer","non_cancer"],
            yticklabels=["cancer","non_cancer"])
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title(f"Confusion Matrix @ thresh={best_thresh}")
plt.tight_layout()
plt.savefig("confusion_matrix_tuned.png", dpi=150)
print("📊 Saved: confusion_matrix_tuned.png")
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_true, y_prob)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
plt.plot([0,1],[0,1],"k--")
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Curve — v2")
plt.legend(); plt.grid(True)
plt.savefig("roc_curve_v2.png", dpi=150)
print("📊 Saved: roc_curve_v2.png")
plt.show()