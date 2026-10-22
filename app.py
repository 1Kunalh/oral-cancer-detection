import os
import uuid
from flask import Flask, render_template, request, jsonify, send_file
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from werkzeug.utils import secure_filename

from db import init_db, add_case, get_cases, update_review
from model_utils import predict_image

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/upload", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    patient_id = request.form.get("patient_id", "unknown")
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only JPG, JPEG, and PNG files are allowed"}), 400

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(image_path)

    prediction, confidence, gradcam = predict_image(image_path)
    case_id = uuid.uuid4().hex[:10]

    add_case(case_id, patient_id, unique_name, image_path, prediction, confidence)

    return jsonify({
        "case_id": case_id,
        "prediction": prediction,
        "confidence": round(float(confidence), 4),
        "heatmap": gradcam
    })

@app.route("/api/cases", methods=["GET"])
def cases():
    return jsonify(get_cases())

@app.route("/api/review/<case_id>", methods=["POST"])
def review_case(case_id):
    data = request.get_json()
    doctor_notes = data.get("doctor_notes", "")
    review_status = data.get("review_status", "Pending")
    update_review(case_id, review_status, doctor_notes)
    return jsonify({"status": "updated", "case_id": case_id})

@app.route("/api/report/<case_id>", methods=["GET"])
def generate_report(case_id):
    cases = get_cases()
    case = next((c for c in cases if c["case_id"] == case_id), None)
    if not case:
        return jsonify({"error": "Case not found"}), 404

    report_path = f"reports/{case_id}.pdf"
    os.makedirs("reports", exist_ok=True)

    pdf = canvas.Canvas(report_path, pagesize=letter)
    pdf.setTitle(f"Oral Cancer Screening Report - {case_id}")
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(70, 760, "AI-Based Oral Cancer Screening System")
    pdf.setFont("Helvetica", 12)
    pdf.drawString(70, 730, f"Case ID: {case_id}")
    pdf.drawString(70, 710, f"Patient ID: {case['patient_id']}")
    pdf.drawString(70, 690, f"Prediction: {case['prediction']}")
    pdf.drawString(70, 670, f"Confidence: {case['confidence']}")
    pdf.drawString(70, 650, f"Review Status: {case['review_status']}")
    pdf.drawString(70, 630, f"Doctor Notes: {case['doctor_notes'] or 'Not provided'}")
    pdf.drawString(70, 610, "Disclaimer: This is an AI-assisted preliminary screening result.")
    pdf.drawString(70, 595, "It is not a confirmed medical diagnosis and requires clinician review.")

    pdf.save()

    return send_file(report_path, as_attachment=True)

if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)