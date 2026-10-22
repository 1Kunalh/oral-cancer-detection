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

- `app/app.py`: Flask web app (uses `oral_cancer_model_v3.keras`)
- `templates/`, `static/`: frontend
- `train_v3.py`: training script
- `gradcam.py`, `evaluate_model.py`: Grad-CAM and evaluation
- `oral_cancer_model_v3.keras`: trained model

## Dataset

This project uses the public **OC-MSI (Oral Cancer Multispectral Imaging) dataset** from Kaggle:
https://www.kaggle.com/datasets/zoya77/oral-cancer-multispectral-imaging-dataset-oc-msi/data

The dataset is **not included** in this repository (medical images, size and licensing).
The trained model (`oral_cancer_model_v3.keras`) is included, so the web app runs without it.
Please follow the dataset's licence and citation requirements on the Kaggle page.

### Retrain the model (optional)

1. Download the dataset from the Kaggle link above.
2. Place the images in the folders provided:

```
real_data/
  train/
    cancer/
    non_cancer/
  val/
    cancer/
    non_cancer/
```

3. Run: `python train_v3.py`
4. Evaluate: `python evaluate_model.py`

## Author

Kunal Hembade
