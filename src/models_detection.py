# Detection function using yolo finetuned model
import os
from ultralytics import YOLO

# Object detection (codes, character, seal) function
def detect_object(image_path, model_path='weights/best.pt', conf=0.25, iou=0.45, object_type=['seal', 'code', 'character']):
    """
    Detect objects in an image using a YOLO model.
    
    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the YOLO model weights.
        conf (float): Confidence threshold for detections.
        iou (float): IoU threshold for non-max suppression.
        
    Returns:
        list: List of detected objects with bounding boxes and labels.
    """
    detections = []
    results_type = []
    if 'seal' in object_type:
        print("Detecting seal...")
        
        model_path_seal = "runs/detect/train_seal_detection/" + model_path
        if not os.path.exists(model_path_seal):
            raise FileNotFoundError(f"Model file not found: {model_path_seal}")
        else:
            model_seal = YOLO(model_path_seal)
            results_seal = model_seal(image_path, conf=conf, iou=iou)
            results_type.append(results_seal)
    
    elif 'code' in object_type:
        print("Detecting code...")
        
        model_path_code = "runs/detect/train_code_detection/" + model_path
        if not os.path.exists(model_path_code):
            raise FileNotFoundError(f"Model file not found: {model_path_code}")
        else:
            model_code = YOLO(model_path_code)
            results_code = model_code(image_path, conf=conf, iou=iou)
            results_type.append(results_code)
    
    elif 'character' in object_type:
        print("Detecting character...")
        
        model_path_character = "runs/detect/train_alphanum_detection/" + model_path
        if not os.path.exists(model_path_character):
            raise FileNotFoundError(f"Model file not found: {model_path_character}")
        else:
            model_character = YOLO(model_path_character)
            results_character = model_character(image_path, conf=conf, iou=iou)
            results_type.append(results_character)
    
    # Extract detections
    for results in results_type:
        if not results:
            print("No detections found.")
            continue
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = result.names[box.cls[0]]
                confidence = box.conf[0]
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'label': label,
                    'confidence': confidence
                })
    
    return detections

# Finetuning function
def finetune_model(dataset, val_data, model_path='yolov8n.pt', epochs=50, batch_size=16, project='runs/detect', name='finetuned_model'):
    """
    Finetune a YOLO model on custom data.
    
    Args:
        dataset (str): Path to data.yaml file.
        val (bool): use validation or not.
        model_path (str): Path to the pre-trained YOLO model.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        project (str): Project directory for saving results.
        name (str): Name of the finetuned model.
        
    Returns:
        None
    """
    if not os.path.exists(dataset):
        raise FileNotFoundError(f"Training data not found: {dataset}")
    
    model = YOLO(model_path)
    model.train(data=dataset, val=val_data, epochs=epochs, batch=batch_size, project=project, name=name)