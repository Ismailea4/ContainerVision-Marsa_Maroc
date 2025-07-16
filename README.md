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
├── src/                  # Source code
├── utils/                # Utility functions
├── notebook/             # Jupyter notebooks
├── data/                 # Data samples (do not push large files)
├── cvat_annotation/      # Annotation scripts and files
├── runs/                 # Model outputs (ignored in git)
├── env/                  # Virtual environment (ignored in git)
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

### OCR and Detection Pipeline

The main function for container OCR and seal detection is in [`src/pipeline.py`](src/pipeline.py):

#### Function: `container_OCR`

```python
def container_OCR(
    image_path,
    model_path='weights/best.pt',
    object_type=['seal', 'code', 'character'],
    conf=0.25,
    iou=0.45,
    display=False
)
```

**Parameters:**

- `image_path` (str): Path to the input image.
- `model_path` (str): Path to the trained YOLO model weights.
- `object_type` (list): List of object types to detect (`'seal'`, `'code'`, `'character'`).
- `conf` (float): Confidence threshold for detections.
- `iou` (float): IoU threshold for non-max suppression.
- `display` (bool): If `True`, displays images with bounding boxes and predictions.

**Returns:**

- `dict` with keys:
  - `'predictions'`: Final image with bounding boxes and predicted text.
  - `'code'`: Dictionary with extracted codes (e.g., container number, type).

#### Example Usage

```python
from src.pipeline import container_OCR

image_path = 'notebook/images/1-155405001-OCR-AS-B01.jpg'
model_path = 'weights/best.pt'

result = container_OCR(
    image_path,
    model_path,
    object_type=['code', 'seal', 'character'],
    conf=0.25,
    iou=0.45,
    display=True  # Set to False to disable image display
)

# Print extracted codes
print("Extracted Codes:", result['code'])

# Save or display the final image with predictions
import cv2
cv2.imwrite('output_with_predictions.jpg', result['predictions'])
```

**Note:**

- Make sure your model weights and image paths are correct.
- The function will display images if `display=True`.

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
