import torch
import torch.nn as nn
from ultralytics import YOLO

from pathlib import Path

import cv2
import os
import numpy as np
from PIL import Image


def adaptive_threshold_and_filter(image_path, min_area=50, window_size=31, window_constant=-1, kernel_size=3, op_iterations=1, display=False):
    """
    Applies adaptive thresholding, area filtering, morphological closing, and YOLO object detection to an input image. 
    Returns the morphologically processed binary image and a list of filtered, sorted bounding rectangles for detected objects.

    Args:
        image_path (str): Path to the input image or the image.
        min_area (int, optional): Minimum area for connected components to be kept. Defaults to 50.
        window_size (int, optional): Window size for adaptive thresholding. Defaults to 31.
        window_constant (int, optional): Constant subtracted from mean in adaptive thresholding. Defaults to -1.
        kernel_size (int, optional): Kernel size for morphological closing. Defaults to 3.
        op_iterations (int, optional): Number of morphological operation iterations. Defaults to 1.

    Returns:
        tuple: (closingImage, boundRect)
            closingImage (np.ndarray): The morphologically closed binary image.
            boundRect (list): List of filtered and sorted bounding rectangles (x, y, w, h) for detected objects.
    """

    # Cheak if the image_path is a path or image
    if isinstance(image_path, str):
        # Load the image from the path
        inputImage = cv2.imread(image_path)
        if inputImage is None:
            raise FileNotFoundError(f"Could not read image from path: {image_path}")
    else:
        # Assume image_path is already an image array
        inputImage = image_path.copy()
    
    inputCopy = inputImage.copy()

    # Convert BGR to grayscale:
    grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

    # Set the adaptive thresholding (gaussian) parameters:
    # Apply the threshold:
    binaryImage = cv2.adaptiveThreshold(
        grayscaleImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, window_size, window_constant
    )

    # Invert if mean value is high (background is white)
    mean_val = np.mean(binaryImage)
    if mean_val > 125:
        binaryImage = cv2.bitwise_not(binaryImage)
    #print(f"Mean value of the binary image: {mean_val}")


    # Perform an area filter on the binary blobs:
    componentsNumber, labeledImage, componentStats, componentCentroids = \
        cv2.connectedComponentsWithStats(binaryImage, connectivity=4)

    # Get the indices/labels of the remaining components based on the area stat
    remainingComponentLabels = [i for i in range(1, componentsNumber) if componentStats[i][4] >= min_area]

    # Filter the labeled pixels based on the remaining labels
    filteredImage = np.where(np.isin(labeledImage, remainingComponentLabels), 255, 0).astype('uint8')


    # Morphological closing:
    maxKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closingImage = cv2.morphologyEx(filteredImage, cv2.MORPH_CLOSE, maxKernel, None, None, op_iterations, cv2.BORDER_REFLECT101)


    # Get each bounding box
    contours, hierarchy = cv2.findContours(closingImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours_poly = [None] * len(contours)
    boundRect = []

    for i, c in enumerate(contours):
        if hierarchy[0][i][3] == -1:
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect.append(cv2.boundingRect(contours_poly[i]))

    # Load YOLO model (update the path if needed)
    model = YOLO('weight_alphanum.pt')

    # Run YOLO detection
    results = model(image_path)  # image_path should be defined earlier

    boundRect = []
    confids = []
    inputCopy = cv2.cvtColor(inputCopy, cv2.COLOR_BGR2RGB)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        for box, conf in zip(boxes, confs):
            if conf >= 0.55:
                x1, y1, x2, y2 = map(int, box[:4])
                w = x2 - x1
                h = y2 - y1
                boundRect.append((x1, y1, w, h))
                confids.append(conf)
                color = (0, 255, 0)
                cv2.rectangle(inputCopy, (x1, y1), (x2, y2), color, 2)
    
    # After you get boundRect and inputImage is loaded
    
    # Convert to numpy array for easier processing
    boundRect_np = np.array(boundRect)
    if len(boundRect_np) > 0:
        widths = boundRect_np[:, 2]
        heights = boundRect_np[:, 3]

        mean_w = np.mean(widths)
        mean_h = np.mean(heights)
        std_w = np.std(widths)
        std_h = np.std(heights)

        # Keep only boxes within a threshold (e.g., 1 std) of the mean width and height
        keep = []
        for box in boundRect:
            w, h = box[2], box[3]
            if (abs(h - mean_h)/mean_h <= 0.4):
                keep.append(box)
        boundRect = keep
    
    # Sort the bounding rectangles based on their position
    h, w = inputImage.shape[:2]
    if w > h:
        # Horizontal: sort by x (left to right)
        if h >= w / 3:
            # Height is at least half the width: split into top and bottom halves
            print("The image is horizontal, splitting into top and bottom halves.")
            top_half = [b for b in boundRect if b[1] < h // 2]
            bottom_half = [b for b in boundRect if b[1] >= h // 2]
            top_half_sorted = sorted(top_half, key=lambda b: b[0])
            bottom_half_sorted = sorted(bottom_half, key=lambda b: b[0])
            boundRect = top_half_sorted + bottom_half_sorted
        else:
            # Standard horizontal: sort all left to right
            boundRect = sorted(boundRect, key=lambda b: b[0])
    else:
        # Vertical: sort by y (top to bottom)
        boundRect = sorted(boundRect, key=lambda b: b[1])


    return closingImage, boundRect, confids  # closingImage is not relevant here, so return None


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
        
        model_path_seal = "weight_seal.pt"
        if not os.path.exists(model_path_seal):
            raise FileNotFoundError(f"Model file not found: {model_path_seal}")
        else:
            model_seal = YOLO(model_path_seal)
            results_seal = model_seal(image_path, conf=conf, iou=iou)
            results_type.append(results_seal)
    
    elif 'code' in object_type:
        print("Detecting code...")
        
        model_path_code = "weight_code.pt"
        if not os.path.exists(model_path_code):
            raise FileNotFoundError(f"Model file not found: {model_path_code}")
        else:
            model_code = YOLO(model_path_code)
            results_code = model_code(image_path, conf=conf, iou=iou)
            results_type.append(results_code)
    
    elif 'character' in object_type:
        print("Detecting character...")
        
        model_path_character = "weight_alphanum.pt"
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
                #print(box.cls, box.conf)
                #print(result.names)
                label = result.names[int(box.cls[0])]
                confidence = box.conf[0]
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'label': label,
                    'confidence': confidence
                })
    
    return detections

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# ResNet-like Model
class ResNetChar(nn.Module):
    def __init__(self, num_classes):
        super(ResNetChar, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(32, 64, stride=2)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride, downsample),
            ResidualBlock(out_channels, out_channels)
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# Load the model for inference
def load_model(model_path, num_classes):
    model = ResNetChar(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def check_digit_verification(code):
    """
    Validates a code by checking if its check digit matches a calculated value.

    The function assigns numerical values to letters, verifies the code format,
    and computes a weighted sum to compare against the check digit.

    Args:
        code (dict): Dictionary containing a 'CN' key with an 11-character string.

    Returns:
        bool: True if the check digit is valid, False otherwise.
    """
    # Create numerical value assigned to each letter of the alphabet using code ascii
    cst = 10
    letter_values = {}
    for i in range(65, 91):
        if (i - 65 + cst) % 11 == 0:
            cst += 1 
        letter_values[chr(i)] = i - 65 + cst    
    
    
    # Verify if the first 4 are letters and the last 7 are digits
    if len(code["CN"]) == 11 and code["CN"][:4].isalpha() and code["CN"][4:].isdigit():
        check_digit = int(code["CN"][-1])
        i = 0
        sum = 0
        for char in code["CN"][:-1]:
            if char.isalpha():
                sum += letter_values[char] * 2**i
            else:
                sum += int(char) * 2**i
            i += 1
        if sum % 11 == 10:
            sum = 0
    
        return check_digit == sum % 11
    return False

def code_verification(code):
    """
    Verify the extracted code.
    
    Args:
        code (dict): Dictionary containing the extracted codes.
        
    Returns:
        bool: True if the code is valid, False otherwise.
    """
    # Example verification logic (to be replaced with actual logic)
    if code["CN"]:
        # Replace '-' by '' in the CN code
        code['CN'] = code['CN'].replace('-', '')
        if len(code['CN']) == 11:
            # Change 0 by O and 1 by I in the first 4 characters 
            code['CN'] = code['CN'][:4].replace('0', 'O') + code['CN'][4:]
            code['CN'] = code['CN'][:4].replace('1', 'I') + code['CN'][4:].replace('I', '1')
            
            if not check_digit_verification(code):
                print("Invalid CN code")
                #code['CN'] = ""
        else:
            print("Invalid CN code length")
    
    if code["TS"]:
        # Replace '-' by '' in the TS code
        code['TS'] = code['TS'].replace('-', '')
        if len(code['TS']) == 4:
            # Change I by 1 in the 4 characters
            code['TS'] = code['TS'].replace('I', '1')

        else:
            print("Invalid TS code length")
    return code

def preprocess_image(img):
    # If not grayscale, convert to grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize to 32x32
    img_resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    # Convert to float32 and scale to [0,1]
    img_resized = img_resized.astype(np.float32) / 255.0
    # Add channel dimension (1, 32, 32)
    img_tensor = torch.from_numpy(img_resized).unsqueeze(0)
    # Add batch dimension (1, 1, 32, 32)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

def container_OCR(image_path, model_path='weights/best.pt', object_type=['seal', 'code', 'character'], conf=0.25, iou=0.45, display=False):
    """
    Detect objects in an image using a specified model and return their bounding boxes and labels.
    
    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the trained model.
        object_type (str): Type of object to detect ('seal', 'code', 'character').
        conf (float): Confidence threshold for detections.
        iou (float): IoU threshold for non-max suppression.
        
    Raises:
        FileNotFoundError: If the model file does not exist.
        
    Returns:
        list: List of detected objects with bounding boxes and labels.
    """
    # Load the image
    image = cv2.imread(image_path)
    
    detections = detect_object(image_path, model_path, conf, iou, ['code'])
    conf = {'CN':1, 'TS':1}

    # draw the bounding boxes on the image
    for detection in detections:
        if detection['label'] == 'seal':
            color = (0, 255, 0)  # Green for seal
        elif detection['label'] == 'code':
            color = (255, 0, 0)  # Blue for code
        elif detection['label'] == 'character': 
            color = (0, 0, 255)  # Red for character
        else:
            color = (255, 255, 0)  # Yellow for unknown
        x_min, y_min, x_max, y_max = detection['bbox']
        cv2.rectangle(image,(x_min, y_min), (x_max, y_max), color, 2)
        #print(f"Detected {detection['label']} at {detection['bbox']}")
        
        cv2.putText(image, detection['label'], (detection['bbox'][0], detection['bbox'][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    

    
    # Remove the bounding boxes from the image
    image = cv2.imread(image_path)
    
    # Crop the image to the bounding boxes
    cropped_images = []
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        cropped_image = image[y1:y2, x1:x2]
        cropped_images.append({
            'label': detection['label'],
            'image': cropped_image
        })
    
    # Apply adaptive thresholding and filtering
    processed_images = []
    for cropped in cropped_images:
        processed_image, boundingChar, confids = adaptive_threshold_and_filter(cropped['image'],display=display)
        processed_images.append({
            'label': cropped['label'],
            'image': processed_image,
            'bounding_boxes': boundingChar
        })
        if cropped['label'] == 'CN':
            conf['CN'] *= np.mean(confids)
        elif cropped['label'] == 'TS':
            conf['TS'] *= np.mean(confids)
    
    
    # Crop the images to bounding character rectangles
    final_images = []
    for processed in processed_images:
        for rect in processed['bounding_boxes']:
            x, y, w, h = rect
            final_image = processed['image'][y:y+h, x:x+w]
            final_images.append({
                'label': processed['label'],
                'image': final_image
            })
        
    # Predict the characters in the final cropped images
    predictions = []
    model_class = ['-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    print(len(model_class), "classes")
    model = load_model("resnet_char_cnn.pth", num_classes=len(model_class))
    code = {'CN':"", 'TS':""}
    for final in final_images:
        # Convert numpy array (OpenCV image) to PIL Image
        img = final['image']
        image_tensor = preprocess_image(img)
        output = model(image_tensor)
        _, predicted = output.max(1)
        char = model_class[predicted.item()]  # Use model_class for mapping
        predictions.append({
            'label': final['label'],
            'character': char,
            'image': final['image']
        })
        #print(f"Predicted character: {char} for label: {final['label']}")
        if final['label'] == 'CN':
            code['CN'] += char
        elif final['label'] == 'TS':
            code['TS'] += char
    
    image = cv2.imread(image_path)

    # Add the text in the original image of bounding boxes
    for detection in detections:
        if detection['label'] == 'seal':
            color = (0, 255, 0)  # Green for seal
        elif detection['label'] == 'code' or detection['label'] == 'TS':
            color = (255, 0, 100)  # Blue for code
        elif detection['label'] == 'character' or detection['label'] == 'CN': 
            color = (100, 0, 255)  # Red for character
        else:
            color = (255, 255, 0)  # Yellow for unknown
        conf[detection['label']] *= detection['confidence']
        x_min, y_min, x_max, y_max = detection['bbox']
        cv2.rectangle(image,(x_min, y_min), (x_max, y_max), color, 2)
        # print(f"Detected {detection['label']} at {detection['bbox']}")
        label_text = f"{code[detection['label']]} ({detection.get('confidence', 0)*100:.2f}%)"
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        # Set the rectangle coordinates
        rect_x1 = x_min
        rect_y1 = y_min - 2*text_height - baseline if y_min - 2*text_height - baseline > 0 else y_min
        rect_x2 = x_min + text_width
        rect_y2 = y_min

        # Draw filled rectangle (background)
        cv2.rectangle(image, (rect_x1, rect_y1), (rect_x2, rect_y2), color, thickness=-1)
        cv2.putText(image,
                    label_text,
                    (detection['bbox'][0], detection['bbox'][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,255,255),
                    2)
    

        
    code = code_verification(code)
    
    detections = { label : {'confidence': float(conf[label]), 'value': code[label]} for label in code if code[label] }
        
    return {
        #'detections': detections,
        #'cropped_images': cropped_images,
        #'processed_images': processed_images,
        #'final_images': final_images,
        'detections': detections,
        'predictions': image
    }
    

def seal_check(detection):
    """
    Check the number of seals.

    Args:
        detection (list): List of detection dictionaries containing label and bounding box.

    Returns:
        bool: True if at least one seal is detected, False otherwise.
        int: Number of seals detected.
    """
    if not detection:
        print("No seal detected")
        return False, None
    num_seals = sum(1 for det in detection if det.get('label') == 'sealed')
    if num_seals > 0:
        print(f"{num_seals} seal(s) detected")
        return True, num_seals
    else:
        print("No seal detected")
        return False, 0


def container_seal(image_path, model_path='weights/best.pt', conf=0.25, iou=0.45, display=False):
    """
    Detect seals in an image using a specified model and return their bounding boxes and labels.
    
    Args:
        image_path (str): Path to the input image or an image.
        model_path (str): Path to the trained model.
        conf (float): Confidence threshold for detections.
        iou (float): IoU threshold for non-max suppression.
        
    Raises:
        FileNotFoundError: If the model file does not exist.
        
    Returns:
        list: List of detected seals with bounding boxes and labels.
    """
    try:
        # Load the image
        image = cv2.imread(image_path)
    except AttributeError:
        image = image_path
    
    detections = detect_object(image_path, model_path, conf, iou, ['seal'])
    
    # Check if seals are detected
    check, num_seals = seal_check(detections)
    seal_num = {'sealed': 0, 'unsealed': 0}
    confids = {'sealed': 1, 'unsealed': 1}
    if num_seals is None:
        print("No seals detected in the image.")
        # Return the original image if no seals are detected
        return {
            'detections': [],
            'predictions': image
        }
    elif not check:
        seal_num['unsealed'] = len(detections) - num_seals
        print(f"There are {seal_num['unsealed']} unseals detected in the image.")
        
    else:
        seal_num['sealed'] = num_seals
        seal_num['unsealed'] = len(detections) - num_seals
        print(f"There are {num_seals} seals detected in the image.")

    # draw the bounding boxes on the image
    for detection in detections:
        if detection['label'] == 'sealed':
            color = (0, 125, 0)  # Green for seal
            confids['sealed'] *= detection['confidence']
        elif detection['label'] == 'unsealed':
            color = (0, 0, 125)  # Red for unsealed
            confids['unsealed'] *= detection['confidence']
        x_min, y_min, x_max, y_max = detection['bbox']
        cv2.rectangle(image,(x_min, y_min), (x_max, y_max), color, 2)
        label_text = f"{detection['label']} ({detection.get('confidence', 0)*100:.2f}%)"
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        # Set the rectangle coordinates
        rect_x1 = x_min
        rect_y1 = y_min - text_height - baseline if y_min - text_height - baseline > 0 else y_min
        rect_x2 = x_min + text_width
        rect_y2 = y_min

        # Draw filled rectangle (background)
        cv2.rectangle(image, (rect_x1, rect_y1), (rect_x2, rect_y2), color, thickness=-1)  # Black background

        # Draw text
        cv2.putText(image, label_text, (x_min, rect_y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        

        
    detections = { label : {'confidence': float(confids[label]), 'value': seal_num[label]} for label in seal_num if seal_num[label] }
    
    return {
        'detections': detections,
        'predictions': image
    }
    
def container_detection(image_path, model_path='weights/best.pt', object_type=['seal', 'code', 'character'], conf=0.25, iou=0.45, display=False):
    """
    Detect objects in an image using a specified model and return their bounding boxes and labels.
    
    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the trained model.
        object_type (str): Type of object to detect ('seal', 'code', 'character').
        conf (float): Confidence threshold for detections.
        iou (float): IoU threshold for non-max suppression.
        
    Raises:
        FileNotFoundError: If the model file does not exist.
        
    Returns:
        list: List of detected objects with bounding boxes and labels.
    """
    
    results = container_OCR(image_path, model_path, object_type, conf, iou, display)
    
    if "seal" in object_type:
        seal_result = container_seal(results['predictions'], model_path, conf, iou, display)
        results['detections'].update(seal_result['detections'])
        results['predictions'] = seal_result['predictions']
        
    return results








from PIL import Image

def detect_container_info(image_path):
    # Your existing detection logic
    # e.g., container_number = ..., iso_code = ..., sealed = ..., unsealed = ...

    # For now, fake example:
    result = {
        "CN": "TEMU1234567",
        "TS": "22G1",
        "sealed": 2,
        "unsealed": 1
    }

    image_with_detections = Image.open(image_path)  # Replace with your detection overlay output

    return result, image_with_detections
