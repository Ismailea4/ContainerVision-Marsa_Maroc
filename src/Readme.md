# src Directory

This folder contains the core source code for the ContainerVision-Marsa_Maroc project. Below is a description of each file and its purpose:

---

## File Overview

- **`__init__.py`**
  - Marks this directory as a Python package.

- **`pipeline.py`**
  - Main pipeline functions for container OCR, seal detection, and object detection.
  - Integrates preprocessing, model inference, and post-processing.
  - Key functions:
    - `container_OCR`: Detects and recognizes container codes and characters.
    - `container_seal`: Detects and classifies seals (sealed/unsealed).
    - `container_detection`: General detection pipeline for containers.

- **`models_detection.py`**
  - Contains model loading and inference utilities.
  - Functions to load YOLO and character recognition models.
  - Provides `detect_object` and `load_model` for use in the pipeline.

- **`data_preparation.py`**
  - Scripts and functions for preparing and labeling data.
  - Includes:
    - Adaptive thresholding and filtering for preprocessing.
    - Functions to create labeled images with bounding boxes and text.
    - Utilities for extracting and saving seal bounding box images.
    - Example: `create_labeled_images`, `extract_and_save_seal_bboxe_image`.

- **Model Weights**
  - `char_cnn.pth`, `advanced_char_cnn.pth`, `advanced_char_cnn2.pth`, `resnet_char_cnn.pth`
    - Pretrained PyTorch models for character recognition.
    - Used by the pipeline for OCR tasks.

---

## Typical Workflow

1. **Data Preparation**
   - Use `data_preparation.py` to preprocess images and generate labeled data for training and evaluation.
   - Example: Draw bounding boxes, create label files, extract seal regions.

2. **Model Loading and Detection**
   - `models_detection.py` provides functions to load YOLO and character CNN models.
   - Detection and recognition are performed using these models.

3. **Pipeline Execution**
   - The main pipeline functions in `pipeline.py` orchestrate the detection, OCR, and seal classification.
   - These functions are called by the main application (`main.py`) or the Flask API (`app.py`).

---

## Example Usage

```python
from src.pipeline import container_OCR, container_seal

result = container_OCR("path/to/image.jpg")
print(result['detections'])

seal_result = container_seal("path/to/image.jpg")
print(seal_result['detections'])
```

---

## Notes

- Do not modify or remove model files (`*.pth`) unless you are updating the trained weights.
- For detailed usage, see the main project [README.md](../README.md).
- For annotation and data conversion scripts, see `data_preparation.py`.

---

For any questions about the code in this folder, please refer to the docstrings in each file or open an issue in