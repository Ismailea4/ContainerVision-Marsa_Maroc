import cv2
import os
from src.data_preparation import adaptive_threshold_and_filter
from src.models_detection import detect_object, load_model
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

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
    
    detections = detect_object(image_path, model_path, conf, iou, object_type)

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
    
    # Show the image with detections
    if display:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    
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
        processed_image, boundingChar = adaptive_threshold_and_filter(cropped['image'],display=display)
        processed_images.append({
            'label': cropped['label'],
            'image': processed_image,
            'bounding_boxes': boundingChar
        })
    
    # Display the processed images
    if display:
        for processed in processed_images:
            plt.imshow(processed['image'], cmap='gray')
            plt.title(f"Processed Image - {processed['label']}")
            plt.axis('off')
            plt.show()
    
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
    model = load_model("./src/resnet_char_cnn.pth", num_classes=len(model_class))
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    code = {'CN':"", 'TS':""}
    for final in final_images:
        # Convert numpy array (OpenCV image) to PIL Image
        img = final['image']
        if len(img.shape) == 2:  # grayscale
            pil_img = Image.fromarray(img)
        else:  # color
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        image_tensor = transform(pil_img).unsqueeze(0)
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
        x_min, y_min, x_max, y_max = detection['bbox']
        cv2.rectangle(image,(x_min, y_min), (x_max, y_max), color, 2)
        # print(f"Detected {detection['label']} at {detection['bbox']}")
        
        cv2.putText(image, code[detection['label']], (detection['bbox'][0], detection['bbox'][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Show the final image with predictions
    if display:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        
    code = code_verification(code)
        
    return {
        #'detections': detections,
        #'cropped_images': cropped_images,
        #'processed_images': processed_images,
        #'final_images': final_images,
        'predictions': image,
        'code': code
    }

# Example usage
if __name__ == "__main__":
    image_path = 'notebook/images/1-155405001-OCR-AS-B01.jpg'
    model_path = 'weights/best.pt'
    result = container_OCR(image_path, model_path,object_type=['code'])
    
    # Print the results
    #print("Detections:", result['detections'])
    #print("Cropped Images:", len(result['cropped_images']))
    #print("Processed Images:", len(result['processed_images']))
    #print("Final Images:", len(result['final_images']))
    #print("Predictions:", result['predictions'])
    print("Code:", result['code'])
    
    # Display the final images with predictions
    """
    for final in result['final_images']:
        plt.imshow(final['image'])
        plt.title(f"Label: {final['label']}")
        plt.show()
    """