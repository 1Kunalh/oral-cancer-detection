from pathlib import Path
from PIL import Image
import numpy as np

base = Path(r"C:\gauravpro\oral_cancer_project\dataset")

for folder in [
    base / "train" / "non_cancer",
    base / "train" / "cancer",
    base / "val" / "non_cancer",
    base / "val" / "cancer",
]:
    folder.mkdir(parents=True, exist_ok=True)

for i in range(3):
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    Image.fromarray(arr).save(base / "train" / "non_cancer" / f"non_{i}.jpg")

for i in range(3):
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    Image.fromarray(arr).save(base / "train" / "cancer" / f"can_{i}.jpg")

for i in range(1):
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    Image.fromarray(arr).save(base / "val" / "non_cancer" / f"non_val_{i}.jpg")

for i in range(1):
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    Image.fromarray(arr).save(base / "val" / "cancer" / f"can_val_{i}.jpg")

print("Dummy dataset created.")
