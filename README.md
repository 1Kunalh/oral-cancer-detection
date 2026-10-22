# Oral Cancer Detection

Deep learning (EfficientNetB0) based oral cancer image classification with a Flask web app and Grad-CAM visualisation.

> **Disclaimer:** For educational and research purposes only. Not a medical diagnosis tool.

## Setup and Run

```bash
git clone https://github.com/1Kunalh/oral-cancer-detection.git
cd oral-cancer-detection
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app\app.py
```

Open http://127.0.0.1:5000

## Structure
- app/app.py: Flask web app (uses oral_cancer_model_v3.keras)
- templates/, static/: frontend
- train_v3.py: training script
- gradcam.py, evaluate_model.py: Grad-CAM and evaluation
- oral_cancer_model_v3.keras: trained model

## Author
Kunal Hembade
