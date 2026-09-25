# 🪖 Helmet Detector — AI-Powered Safety Monitoring

<img width="1917" height="1015" alt="image" src="https://github.com/user-attachments/assets/31ce6926-ebc2-480d-a369-050c24c2d8ab" />


A computer-vision app that detects whether workers are wearing safety helmets, built with **YOLO (Ultralytics)** and a **Streamlit** dashboard UI.

---

## Features

- **Multiple input modes**: upload an image, upload a video, take a live webcam snapshot, or run a continuous live webcam feed
- **Person-level compliance matching**: links each detected person to the nearest helmet/no-helmet detection near their head, instead of just listing raw boxes
- **Configurable thresholds**: confidence and IoU (NMS) sliders, class-agnostic NMS to avoid duplicate overlapping boxes on the same head
- **Debug mode**: inspect the model's raw, unfiltered output (including a near-zero-confidence pass) to diagnose detection issues
- **Session dashboard**: running totals of detections, with/without-helmet counts, and last-scan time, with deltas vs. your previous scan
- **CSV export** of all detections for a given run

---

## Project structure

```
.
├── app.py                      # Streamlit app (main entry point)
├── requirements.txt            # Python dependencies
├── best.pt                     # Trained model weights (not committed — see .gitignore)
├── notebooks/
│   ├── VOC2028_YOLOv8_Training.ipynb          # Original VOC-format training (Colab)
│   ├── Helmet_Detection_YOLO11_Retrain.ipynb  # Retraining on Roboflow dataset (Colab)
│   └── ...
├── select_test_images.py       # Pulls diverse test images from the dataset for manual QA
├── select_more_no_helmet.py    # Pulls additional no-helmet test images
└── README.md
```

---

## Setup

1. **Clone the repo and install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Place your trained model weights** at the path configured in `app.py`'s `get_model()` function, or upload them via the app's sidebar/Model Settings panel at runtime.

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

---

## Model

- **Architecture**: YOLO11 (Ultralytics), fine-tuned from COCO-pretrained weights
- **Classes**: `helmet`, `head` (unprotected head, i.e. "no helmet"), `person`
- **Training data**: [Hard Hat Workers dataset](https://universe.roboflow.com/joseph-nelson/hard-hat-workers) (Northeastern University, ~7,000 images), and/or the `bac_hien_construction_safety_2024` Roboflow dataset — see the training notebooks for the exact pipeline used, including class-balance checks and a fix for a stray `null` class some Roboflow exports include.

### Validation results (example run)

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Helmet | 0.924 | 0.829 | 0.896 | 0.547 |
| No-Helmet (head) | 0.860 | 0.656 | 0.746 | 0.509 |
| Person | 0.770 | 0.680 | 0.702 | 0.465 |

*(Numbers will vary depending on which dataset/training run you use — regenerate this table with `model.val()` after training your own weights.)*

---

## Known limitations

- **Domain-specific**: the model is trained on real construction-site-style imagery. It performs noticeably worse on stylistically different inputs (e.g. polished stock photography, studio portraits, extreme close-ups) — this is expected behavior for a model trained on a specific visual domain, not a bug.
- **`Person` class**: depending on which training run/dataset you use, `person`-class accuracy may be weaker than `helmet`/`head`. If your model's `person` mAP is poor, leave "Person-level compliance matching" **off** in Advanced Settings — it will otherwise show most people as "Unknown" rather than a real compliance status.
- **Continuous Live Feed** mode requires `streamlit-webrtc`, browser camera permission, and runs inference on a frame-skip interval (not every frame) since CPU inference can't keep up with real-time video otherwise. It does not currently track live aggregate stats — only Upload Image, Upload Video, and Live Snapshot modes update the dashboard's running totals.

---

## Retraining

The `notebooks/` folder contains Colab notebooks for retraining from scratch, including:
- VOC-XML → YOLO label conversion (for VOC-format datasets like the original VOC2028 set)
- Roboflow dataset download + a class-balance check to catch imbalanced classes before training
- A cleanup step to strip stray `null` classes some Roboflow exports include
- Training with augmentation tuned for small-object detection (helmets are small relative to the full frame) and higher input resolution

---

## License

Add your license here (e.g. MIT).
