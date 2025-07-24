# ContainerVision-Marsa_Maroc

AI solution for automated container ID and seal recognition at Marsa Maroc terminals.

## Project Overview

This project leverages OCR and computer vision to automatically extract container shipping information and detect seals from images. It is designed for deployment at Marsa Maroc terminals to improve efficiency and accuracy in container handling.

## Folder Structure

```
.
├── .gitignore
├── README.md
├── requirements.txt
├── yolov8n.pt
├── ocr_fast.py
├── prepare_yolo.py
├── main.py                # Main entry point for detection
├── src/                   # Source code (pipelines, models, utils)
├── utils/                 # Utility functions
├── notebook/              # Jupyter notebooks
├── data/                  # Data samples (do not push large files)
├── cvat_annotation/       # Annotation scripts and files
├── runs/                  # Model outputs (ignored in git)
├── env/                   # Virtual environment (ignored in git)
```

## Getting Started

### Prerequisites

- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/)
- [YOLOv8](https://docs.ultralytics.com/)
- (Optional) [virtualenv](https://virtualenv.pypa.io/en/latest/)

### Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/yourusername/ContainerVision-Marsa_Maroc.git
   cd ContainerVision-Marsa_Maroc
   ```

2. **Create and activate a virtual environment (recommended):**

   ```sh
   python -m venv env
   .\env\Scripts\activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

### Data

- Place your raw and processed data in the `data/` directory.
- Annotation files and scripts are in `cvat_annotation/`.
- **Note:** Large datasets and model weights should not be pushed to GitHub. Use `.gitignore` to exclude them.

---

## Usage

### Run Detection from Command Line

The main entry point for detection and OCR is via the `main.py` script, which uses the `container_detection` pipeline from `src/pipeline.py`.

#### Example Command

```sh
python main.py --image data/test/1-153655001-OCR-RF-D01.jpg --model weights/best.pt --object_type code seal character --display --output result.jpg
```

**Arguments:**

- `--image`: Path to the input image or directory (required)
- `--model`: Path to YOLO model weights (default: `weights/best.pt`)
- `--char_model`: Path to character CNN model (default: `char_cnn.pth`)
- `--object_type`: List of object types to detect (`code`, `seal`, `character`)
- `--conf`: Confidence threshold (default: `0.25`)
- `--iou`: IoU threshold (default: `0.45`)
- `--display`: Show the image with predictions
- `--output`: Path to save the output image (default: `output_with_predictions.jpg`)

**Example Output:**

```
Extracted Detections: {'CN': {'confidence': 0.98, 'value': 'ABCD1234567'}, 'TS': {'confidence': 0.95, 'value': '1234'}, 'sealed': {'confidence': 0.92, 'value': 2}}
Output image saved to result.jpg
```

---

### Using the Pipeline Functions Directly

You can use the main detection and seal functions in your own Python scripts.  
The pipeline returns:

- `'detections'`: a dictionary with detection types as keys and a dictionary of confidence and value as values.
- `'predictions'`: the annotated image (with bounding boxes, code, and confidence displayed).

#### Example: Container OCR and Detection

```python
from src.pipeline import container_detection

image_path = 'data/test/1-153655001-OCR-RF-D01.jpg'
model_path = 'weights/best.pt'

result = container_detection(
    image_path=image_path,
    model_path=model_path,
    object_type=['code', 'seal'],
    conf=0.25,
    iou=0.45,
    display=True
)

print("Detections:", result['detections'])

import cv2
cv2.imwrite('output_with_predictions.jpg', result['predictions'])
```

#### Example: Seal Detection Only

```python
from src.pipeline import container_seal

image_path = 'data/test/1-153655001-OCR-RF-D01.jpg'
model_path = 'weights/best.pt'

seal_result = container_seal(
    image_path=image_path,
    model_path=model_path,
    conf=0.25,
    iou=0.45,
    display=True
)

print("Seal Detections:", seal_result['detections'])
```

---

### Flask API

A simple Flask API is provided in `app.py` to allow image upload and return the annotated image and detected codes.

**Example usage:**

1. Start the server:

   ```sh
   python app.py
   ```

2. Open [http://localhost:5000/](http://localhost:5000/) in your browser to use the web interface.

3. Or use curl:
   ```sh
   curl -F "image=@path/to/your/image.jpg" http://localhost:5000/detect --output result.png
   curl -F "image=@path/to/your/image.jpg" http://localhost:5000/detect_json
   ```

---

### Other Scripts

- **OCR and Detection:**  
  Main scripts for OCR and YOLO preparation are in the root and `src/` directories.
  ```sh
  python ocr_fast.py
  python prepare_yolo.py
  ```
- **Annotation:**  
  Use scripts in `cvat_annotation/` for annotation conversion and visualization.
- **Notebooks:**  
  Explore and test models in the `notebook/` directory.

### Training

1. Prepare your dataset in YOLO format (see `prepare_yolo.py`).
2. Train your model using YOLOv8 or your preferred framework.

### Results

- Model outputs and logs are stored in the `runs/` directory.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](LICENSE) (or your chosen license)

## Acknowledgements

- Marsa Maroc
- Ultralytics YOLO
- OpenCV, Tesseract OCR

---

_For more details, see the code and comments in each directory._
