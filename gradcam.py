import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os

model = tf.keras.models.load_model("oral_cancer_model_v3.keras")
IMG_SIZE = (224, 224)

# ---------- Model components ----------
aug  = model.get_layer('sequential')
base = model.get_layer('efficientnetb0')

# ---------- Graph explicit rebuild (Keras 3 nested issue fix) ----------
inp = tf.keras.Input(shape=(224, 224, 3))
x = aug(inp)
conv_feat = base(x, training=False)          # (None, 7, 7, 1280) — spatial features
x2 = model.get_layer('global_average_pooling2d')(conv_feat)
x2 = model.get_layer('dropout')(x2, training=False)
x2 = model.get_layer('dense')(x2)
x2 = model.get_layer('dropout_1')(x2, training=False)
preds = model.get_layer('dense_1')(x2)

grad_model = tf.keras.Model(inputs=inp, outputs=[conv_feat, preds])
print("✅ grad_model built")
print(f"   Input: {grad_model.input_shape}")
print(f"   Outputs: {grad_model.output_shape}\n")
# ---------- Grad-CAM ----------
def make_gradcam(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)       # ⚠️ NO /255 — [0, 255] range
    arr = np.expand_dims(arr, 0)

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(arr, training=False)
        loss = 1.0 - preds[:, 0]                 # cancer prob = 1 - sigmoid output

    grads = tape.gradient(loss, conv_out)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    if tf.math.reduce_max(heatmap) > 0:
        heatmap = heatmap / tf.math.reduce_max(heatmap)

    return heatmap.numpy(), float(preds[0][0])

# ---------- 4 samples visualize ----------
cancer_dir    = "real_data/val/cancer"
noncancer_dir = "real_data/val/non_cancer"
cancer_imgs    = sorted(os.listdir(cancer_dir))[:2]
noncancer_imgs = sorted(os.listdir(noncancer_dir))[:2]

all_imgs = [(cancer_dir, f, "cancer")     for f in cancer_imgs] + \
           [(noncancer_dir, f, "non_cancer") for f in noncancer_imgs]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))

for idx, (folder, fname, label) in enumerate(all_imgs):
    img_path = os.path.join(folder, fname)
    heatmap, prob = make_gradcam(img_path)

    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hm_resized = cv2.resize(heatmap, IMG_SIZE)
    hm_color = cv2.applyColorMap(np.uint8(255 * hm_resized), cv2.COLORMAP_JET)
    hm_rgb = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    overlayed = cv2.addWeighted(img_rgb, 0.55, hm_rgb, 0.45, 0)

    row, col = idx // 2, (idx % 2) * 2

    axes[row, col].imshow(img_rgb)
    axes[row, col].set_title(f"Original ({label})")
    axes[row, col].axis("off")

    pred_label = "cancer" if prob < 0.5 else "non_cancer"
    match = "✅" if pred_label == label else "❌"
    axes[row, col + 1].imshow(overlayed)
    axes[row, col + 1].set_title(f"Grad-CAM {match}\npred: {pred_label}, p={prob:.2f}")
    axes[row, col + 1].axis("off")

plt.tight_layout()
plt.savefig("gradcam_v3.png", dpi=150)
print("📊 Saved: gradcam_v3.png")
plt.show()