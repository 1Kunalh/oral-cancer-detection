import os
import numpy as np
import tensorflow as tf
import cv2
from datetime import datetime
from flask import Flask, request, render_template, url_for, session, redirect
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MODEL_PATH = os.path.join(PROJECT_DIR, "oral_cancer_model_v3.keras")
IMG_SIZE = (224, 224)
ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "oral-cancer-ai-secret-key-2026"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# ---------- Model load ----------
print("🔄 Model loading...")
model = tf.keras.models.load_model(MODEL_PATH)
aug  = model.get_layer("sequential")
base = model.get_layer("efficientnetb0")

inp = tf.keras.Input(shape=(224, 224, 3))
x = aug(inp)
conv_feat = base(x, training=False)
x2 = model.get_layer("global_average_pooling2d")(conv_feat)
x2 = model.get_layer("dropout")(x2, training=False)
x2 = model.get_layer("dense")(x2)
x2 = model.get_layer("dropout_1")(x2, training=False)
preds = model.get_layer("dense_1")(x2)
grad_model = tf.keras.Model(inputs=inp, outputs=[conv_feat, preds])
print("✅ Model ready!")

# ---------- Helpers ----------
def allowed_file(fn):
    return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def make_gradcam(img_path, save_path):
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, 0)
    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(arr, training=False)
        loss = 1.0 - pred[:, 0]
    grads = tape.gradient(loss, conv_out)[0]
    pooled = tf.reduce_mean(grads, axis=(0, 1))
    hm = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
    hm = tf.maximum(hm, 0)
    if tf.math.reduce_max(hm) > 0:
        hm = hm / tf.math.reduce_max(hm)
    img_bgr = cv2.imread(img_path)
    img_bgr = cv2.resize(img_bgr, IMG_SIZE)
    hm = cv2.resize(hm.numpy(), IMG_SIZE)
    hm_color = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.55, hm_color, 0.45, 0)
    cv2.imwrite(save_path, overlay)


def predict(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    arr = np.expand_dims(tf.keras.utils.img_to_array(img), 0)
    prob = float(model.predict(arr, verbose=0)[0][0])
    if prob < 0.5:
        return "cancer", (1 - prob) * 100, prob
    return "non_cancer", prob * 100, prob


def get_doctor_review(label, confidence):
    """Simulate doctor's review based on AI confidence."""
    if label == "cancer":
        if confidence >= 90:
            return {
                "severity": "HIGH",
                "color": "#dc2626",
                "assessment": (
                    "The AI model demonstrates high-confidence features consistent "
                    "with malignant transformation of the oral mucosa. The heatmap "
                    "localization corresponds to the lesion area, supporting the "
                    "diagnostic suspicion."
                ),
                "action": (
                    "IMMEDIATE clinical correlation is advised. An incisional biopsy "
                    "with histopathological examination is strongly recommended. "
                    "Referral to an oral oncologist / head-and-neck surgeon should "
                    "be initiated without delay."
                ),
            }
        elif confidence >= 75:
            return {
                "severity": "MODERATE",
                "color": "#ea580c",
                "assessment": (
                    "The AI model has flagged suspicious features suggestive of "
                    "possible malignancy. However, moderate confidence warrants "
                    "clinical confirmation before any definitive conclusion."
                ),
                "action": (
                    "Clinical examination by an oral medicine specialist is advised "
                    "within 1-2 weeks. Consider biopsy if the lesion persists or "
                    "appears clinically suspicious."
                ),
            }
        else:
            return {
                "severity": "LOW-MODERATE",
                "color": "#d97706",
                "assessment": (
                    "AI analysis indicates some features that may be malignant, but "
                    "with low-moderate confidence. This is not definitive and requires "
                    "proper clinical evaluation."
                ),
                "action": (
                    "Close clinical follow-up is recommended. Re-evaluate in 2 weeks. "
                    "Biopsy may be considered if symptoms worsen."
                ),
            }
    else:  # non_cancer
        if confidence >= 90:
            return {
                "severity": "NONE DETECTED",
                "color": "#16a34a",
                "assessment": (
                    "No malignant features were detected by AI analysis. Findings "
                    "are consistent with healthy oral mucosa or a benign lesion. "
                    "No suspicious transformations observed."
                ),
                "action": (
                    "Routine oral examination every 6-12 months is advised. "
                    "Maintain good oral hygiene. Report any new or changing "
                    "symptoms immediately."
                ),
            }
        elif confidence >= 75:
            return {
                "severity": "LOW",
                "color": "#65a30d",
                "assessment": (
                    "Likely benign. AI did not detect strong features of malignancy, "
                    "though moderate confidence suggests continued monitoring."
                ),
                "action": (
                    "No immediate intervention needed. Monitor for any changes. "
                    "Routine follow-up advised. Consult doctor if symptoms develop."
                ),
            }
        else:
            return {
                "severity": "LOW (INCONCLUSIVE)",
                "color": "#ca8a04",
                "assessment": (
                    "Low suspicion of malignancy based on AI, but confidence is "
                    "moderate. Clinical correlation is recommended to rule out "
                    "early-stage lesions."
                ),
                "action": (
                    "Clinical examination advised for definitive assessment. "
                    "Follow-up if symptoms persist or worsen."
                ),
            }


def get_patient_summary(label, confidence):
    """Simple, patient-friendly explanation."""
    if label == "cancer":
        return (
            f"AI ने तुमच्या तोंडाच्या फोटोची तपासणी केली आणि त्यात कर्करोगाची "
            f"(cancer) शक्यता दिसली आहे (विश्वास: {confidence:.1f}%). "
            f"कृपया लगेच डॉक्टरांना भेटा. हे अंतिम निदान नाही — डॉक्टर "
            f"तपासणी करून आवश्यक असल्यास biopsy करतील. लवकर निदान झाल्यास "
            f"उपचार अधिक यशस्वी होतात."
        )
    else:
        return (
            f"AI ने तुमच्या तोंडाच्या फोटोची तपासणी केली आणि त्यात कर्करोगाची "
            f"कोणतीही लक्षणे आढळली नाहीत (विश्वास: {confidence:.1f}%). "
            f"तुमचे तोंडाचे ऊतक सामान्य किंवा सौम्य (benign) दिसत आहेत. "
            f"तरीही दर ६-१२ महिन्यांनी तपासणी करा. जर काही नवीन लक्षणे "
            f"(न भरणारी जखम, पांढरे/लाल ठिपके, गाठ) दिसल्यास त्वरित डॉक्टरांना भेटा."
        )


# ---------- Routes ----------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return render_template("index.html", error="File select kara!")
        if not allowed_file(file.filename):
            return render_template("index.html", error="Fakt PNG/JPG/JPEG/BMP allowed!")

        filename = secure_filename(file.filename)
        orig_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(orig_path)

        gradcam_name = "gradcam_" + filename
        gradcam_path = os.path.join(UPLOAD_FOLDER, gradcam_name)
        make_gradcam(orig_path, gradcam_path)

        label, conf, prob = predict(orig_path)

        # Session madhe save kara (report sathi)
        session["prediction"] = {
            "report_id": "OCR-" + datetime.now().strftime("%Y%m%d%H%M%S"),
            "timestamp": datetime.now().strftime("%d-%m-%Y %H:%M"),
            "patient_name": request.form.get("patient_name", "").strip() or "Anonymous",
            "patient_age": request.form.get("patient_age", "").strip() or "N/A",
            "patient_gender": request.form.get("patient_gender", "").strip() or "N/A",
            "label": label,
            "confidence": round(conf, 2),
            "prob": round(prob, 4),
            "original": filename,
            "gradcam": gradcam_name,
        }

        result = {
            "label": label,
            "label_display": "Cancer" if label == "cancer" else "Non-Cancer",
            "confidence": f"{conf:.2f}",
            "prob_raw": f"{prob:.4f}",
            "original": url_for("static", filename=f"uploads/{filename}"),
            "gradcam": url_for("static", filename=f"uploads/{gradcam_name}"),
            "is_cancer": label == "cancer",
        }

    return render_template("index.html", result=result)


@app.route("/report")
def report():
    data = session.get("prediction")
    if not data:
        return redirect("/")
    review = get_doctor_review(data["label"], data["confidence"])
    summary = get_patient_summary(data["label"], data["confidence"])
    return render_template("report.html", data=data, review=review, summary=summary)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)