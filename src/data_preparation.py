import os
import shutil
import random
from collections import defaultdict
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Function to organize dataset for a specific interface, ensuring that each image ID has all required angles.

def organise_dataset_for_interface(image_folder, output_folder, required_angles=None, sample_count=10):
    if required_angles is None:
        required_angles = {'LF', 'RF', 'LB', 'AH', 'AS'}
    images_by_id = defaultdict(dict)
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg'):
            parts = filename.split('-')
            if len(parts) >= 4:
                img_id = parts[1]
                angle = parts[3].split('.')[0]
                images_by_id[img_id][angle] = filename
    valid_ids = [img_id for img_id, angles in images_by_id.items() if required_angles.issubset(angles.keys())]
    sample_ids = valid_ids[:sample_count]
    os.makedirs(output_folder, exist_ok=True)
    for img_id in sample_ids:
        id_folder = os.path.join(output_folder, img_id)
        os.makedirs(id_folder, exist_ok=True)
        for angle in required_angles:
            src = os.path.join(image_folder, images_by_id[img_id][angle])
            dst = os.path.join(id_folder, images_by_id[img_id][angle])
            shutil.copy(src, dst)

def is_iso_code(text):
    return len(text) == 4

def is_owner_code(text):
    return len(text) == 11 and text[0:4].isalpha() and text[4:11].isdigit()

# Convert labels from the original format to YOLO format and split into train/val sets (only case of code detection) !!!
def convert_labels_to_yolo(images_dir, labels_dir, output_dir):
    """
    Converts bounding box labels from a custom format to YOLO format, splitting the dataset into training and validation sets.

    Creates necessary directory structure, processes label files, converts bounding box coordinates, assigns class IDs based on label type, and copies images and labels to the appropriate folders. Also prepares a data.yaml configuration for YOLO training.

    Args:
        images_dir (str): Path to the directory containing input images.
        labels_dir (str): Path to the directory containing label files.
        output_dir (str): Path to the output directory for YOLO-formatted data.
    """
    os.makedirs(f'{output_dir}/images/train', exist_ok=True)
    os.makedirs(f'{output_dir}/images/val', exist_ok=True)
    os.makedirs(f'{output_dir}/labels/train', exist_ok=True)
    os.makedirs(f'{output_dir}/labels/val', exist_ok=True)
    all_files = []
    for filename in os.listdir(labels_dir):
        label_path = os.path.join(labels_dir, filename)
        with open(label_path, 'r') as f:
            lines = f.readlines()
        if 1 <= len(lines) <= 2:
            all_files.append(filename.replace('.txt', ''))
    random.shuffle(all_files)
    split_idx = int(0.8 * len(all_files))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    def process_files(file_list, split):
        for file_id in file_list:
            image_path = os.path.join(images_dir, file_id + '.jpg')
            label_path = os.path.join(labels_dir, file_id + '.txt')
            img = Image.open(image_path)
            img_w, img_h = img.size
            yolo_label_lines = []
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) != 5:
                        continue
                    x_min, y_min, x_max, y_max, text = parts
                    x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
                    x_center = (x_min + x_max) / 2 / img_w
                    y_center = (y_min + y_max) / 2 / img_h
                    width = (x_max - x_min) / img_w
                    height = (y_max - y_min) / img_h
                    if is_owner_code(text):
                        class_id = 0
                    elif is_iso_code(text):
                        class_id = 1
                    else:
                        continue
                    yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    yolo_label_lines.append(yolo_line)
            with open(f"{output_dir}/labels/{split}/{file_id}.txt", 'w') as f:
                f.write('\n'.join(yolo_label_lines))
            shutil.copy(image_path, f"{output_dir}/images/{split}/{file_id}.jpg")
    process_files(train_files, 'train')
    process_files(val_files, 'val')
    
    # Add data.yaml file
    data_yaml = {
        'path': output_dir,
        'train': 'images/train',
        'val': 'images/val',
        'nc': 2,
        'names': ['owner_code', 'iso_code']
    }

def save_non_processed_images(images_dir, train_files, val_files, output_dir):
    non_processed_files = set([name.replace(".jpg","") for name in os.listdir(images_dir)]) - set(train_files + val_files)
    non_processed_dir = os.path.join(output_dir, 'images', 'non_processed')
    os.makedirs(non_processed_dir, exist_ok=True)
    for file_id in non_processed_files:
        src_image_path = os.path.join(images_dir, file_id + ".jpg")
        if os.path.exists(src_image_path):
            shutil.copy(src_image_path, non_processed_dir)

def create_labeled_images(image_dir, labeled_image_dir, label_dir):
    """
    Creates labeled images by drawing bounding boxes and classifying text from label files.

    For each image in the input directory, checks for a corresponding label file, draws bounding boxes and text labels on the image, and saves both the labeled image and a new label file with class IDs in the output directory.

    Args:
        image_dir (str): Path to the directory containing image subdirectories.
        labeled_image_dir (str): Path to the directory where labeled images and label files will be saved.
        label_dir (str): Path to the directory containing label files.

    """
    os.makedirs(labeled_image_dir, exist_ok=True)
    for img_id in os.listdir(image_dir):
        for image in os.listdir(os.path.join(image_dir, img_id)):
            image_path = os.path.join(image_dir, img_id, image)
            image_no_ext = os.path.splitext(image)[0]
            label_path = os.path.join(label_dir, image_no_ext + '.txt')
            if os.path.exists(label_path):
                labeled_image = cv2.imread(image_path)
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    label_text = ""
                    for line in lines:
                        parts = line.strip().split(',')
                        if len(parts) != 5:
                            continue
                        x_min, y_min, x_max, y_max, text = parts
                        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
                        cv2.rectangle(labeled_image, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)
                        cv2.putText(labeled_image, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
                        if len(text) == 11:
                            class_id = 0
                        elif len(text) == 4:
                            class_id = 1
                        else:
                            class_id = -1
                        label_text += f"{class_id} {text} {x_min} {y_min} {x_max} {y_max}\n"
                os.makedirs(os.path.join(labeled_image_dir, img_id), exist_ok=True)
                label_file_path = os.path.join(labeled_image_dir, img_id, f"{image_no_ext}.txt")
                with open(label_file_path, 'w') as label_file:
                    label_file.write(label_text.strip())
                labeled_image_save_path = os.path.join(labeled_image_dir, img_id, f"{image_no_ext}.jpg")
                cv2.imwrite(labeled_image_save_path, labeled_image)

def crop_labeled_regions(image_dir, labeled_image_dir):
    for img_id in os.listdir(labeled_image_dir):
        id_folder = os.path.join(labeled_image_dir, img_id)
        if not os.path.isdir(id_folder):
            continue
        for image in os.listdir(id_folder):
            if not image.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            image_no_ext = os.path.splitext(image)[0]
            label_path = os.path.join(id_folder, image_no_ext + '.txt')
            labeled_image_path = os.path.join(id_folder, image)
            image_path = os.path.join(image_dir, image)
            if not os.path.exists(label_path):
                continue
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image {image_path}")
                continue
            out_subfolder = os.path.join(labeled_image_dir, img_id, image_no_ext)
            os.makedirs(out_subfolder, exist_ok=True)
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for idx, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) < 6:
                        continue
                    class_id = parts[0]
                    x_min = int(parts[2])
                    y_min = int(parts[3])
                    x_max = int(parts[4])
                    y_max = int(parts[5])
                    cropped_img = img[y_min:y_max, x_min:x_max]
                    crop_name = f"{image_no_ext}_crop_{idx}_class{class_id}.jpg"
                    crop_path = os.path.join(out_subfolder, crop_name)
                    cv2.imwrite(crop_path, cropped_img)



# Cropping and image reconstitution functions

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

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
    model = YOLO('runs/detect/train_alphanum_detection/weights/best.pt')

    # Run YOLO detection
    results = model(image_path)  # image_path should be defined earlier

    boundRect = []
    inputCopy = cv2.cvtColor(inputCopy, cv2.COLOR_BGR2RGB)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        for box, conf in zip(boxes, confs):
            if conf >= 0.75:
                x1, y1, x2, y2 = map(int, box[:4])
                w = x2 - x1
                h = y2 - y1
                boundRect.append((x1, y1, w, h))
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

    # Visualize the bounding boxes
    if display:
        plt.imshow(inputCopy)
        plt.title("YOLO Bounding Boxes Detected")
        plt.axis('off')
        plt.show()

    return closingImage, boundRect  # closingImage is not relevant here, so return None

def reconstitute_characters_with_padding(closingImage, boundRect, space=10, top_pad=5, bottom_pad=5, left_pad=2, right_pad=2, save_path="reconstituted_image.png"):
    """
    Crops character regions from an image using bounding rectangles, pads each character to uniform height, and concatenates them horizontally with black borders and configurable padding. Displays and saves the resulting image, and returns the final padded image array.

    Args:
        closingImage (np.ndarray): Grayscale image containing characters.
        boundRect (list): List of bounding rectangles for each character (x, y, w, h).
        space (int, optional): Width of black space between characters. Default is 10.
        top_pad (int, optional): Padding at the top of the final image. Default is 5.
        bottom_pad (int, optional): Padding at the bottom of the final image. Default is 5.
        left_pad (int, optional): Padding at the left of the final image. Default is 2.
        right_pad (int, optional): Padding at the right of the final image. Default is 2.
        save_path (str, optional): Path to save the reconstituted image. Default is "reconstituted_image.png".

    Returns:
        np.ndarray: The final padded and concatenated image array.
    """

    import numpy as np
    import cv2
    import matplotlib.pyplot as plt

    
    # Condition to check if the image is vertical
    #if closingImage.shape[0] > closingImage.shape[1]:
    #print("The image is vertical, resizing to horizontal orientation.")

    # Crop character images (reverse order as in your code)
    cropped_images = []
    for i in range(len(boundRect)):
        x, y, w, h = boundRect[i]
        croppedImg = closingImage[y:y + h, x:x + w]
        cropped_images.append(croppedImg)

    # Find the maximum height among all cropped images
    max_height = max(img.shape[0] for img in cropped_images)

    # Resize all images to the same height (pad with black if needed)
    cropped_images_padded = []
    for img in cropped_images:
        h, w = img.shape
        if h < max_height:
            pad = np.zeros((max_height - h, w), dtype=img.dtype)
            img_padded = np.vstack([img, pad])
        else:
            img_padded = img
        cropped_images_padded.append(img_padded)

    # Create black space image
    black_space = np.zeros((max_height, space), dtype=np.uint8)

    # Concatenate images with black space in between
    result = cropped_images_padded[0]
    for img in cropped_images_padded[1:]:
        result = np.hstack([result, black_space, img])

    # Add black space at the top, bottom, left, and right
    result_padded = np.pad(
        result,
        pad_width=((top_pad, bottom_pad), (left_pad, right_pad)),
        mode='constant',
        constant_values=0
    )

    # Show the result
    plt.imshow(result_padded, cmap='gray')
    plt.axis('off')
    plt.title('Reconstituted Image with Black Borders and Padding')
    plt.show()

    # Save the reconstituted image with borders and padding
    cv2.imwrite(save_path, result_padded)
    print(f"Reconstituted image saved as '{save_path}'")

    return result_padded
"""   
    else:
        print("The image is horizontal")
        # Show the original closing image
        plt.imshow(closingImage, cmap='gray')
        plt.axis('off')
        plt.title('Original Closing Image')
        plt.show()
        
        # Save the original closing image
        cv2.imwrite(save_path, closingImage)
        print(f"Original closing image saved as '{save_path}'")
        return closingImage
"""

import os
import cv2

def process_and_reconstitute_all(labeled_samples_dir="labeled_samples"):
    """
    For each id and each labeled image subfolder, reconstitute all crops and save the result.
    """
    for img_id in os.listdir(labeled_samples_dir):
        id_folder = os.path.join(labeled_samples_dir, img_id)
        if not os.path.isdir(id_folder):
            continue
        for labeled_image_subfolder in os.listdir(id_folder):
            subfolder_path = os.path.join(id_folder, labeled_image_subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            # Collect all crop images in this subfolder
            crop_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and '_crop_' in f]
            if not crop_files:
                continue
            # Sort crop files by index in filename for consistent order
            crop_files.sort(key=lambda x: int(x.split('_crop_')[1].split('_')[0]))
            # For each crop, apply adaptive threshold and get bounding rects
            for idx, crop_file in enumerate(crop_files):
                crop_path = os.path.join(subfolder_path, crop_file)
                # Use adaptive threshold and filter to get bounding boxes
                closingImage, boundRect = adaptive_threshold_and_filter(crop_path)
                # Reconstitute the character(s) in this crop
                if boundRect:
                    # Extract class id from filename
                    class_id = crop_file.split('class')[-1].split('.')[0]
                    save_name = f"{labeled_image_subfolder}_reconst_{idx}_class{class_id}.jpg"
                    save_path = os.path.join(subfolder_path, save_name)
                    reconstitute_characters_with_padding(closingImage, boundRect, save_path=save_path)
                    print(f"Saved: {save_path}")

# Run the process
#process_and_reconstitute_all("labeled_samples")


# Semi_supervised model for alphanum detection and saving YOLO labels
# This function will display the bounding boxes and ask the user to accept or skip them.

import os
import cv2
import matplotlib.pyplot as plt

def save_yolo_labels(image_path, boundRect, save_dir):
    """
    Save YOLO-format labels for the bounding boxes.
    """
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    label_lines = []
    for rect in boundRect:
        x, y, bw, bh = rect
        # YOLO format: class x_center y_center width height (all normalized)
        x_center = (x + bw / 2) / w
        y_center = (y + bh / 2) / h
        width = bw / w
        height = bh / h
        class_id = 0  # You can change this if you want to assign different classes
        label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    # Save label file
    base = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(save_dir, base + ".txt")
    with open(label_path, "w") as f:
        f.write("\n".join(label_lines))
    # Save image
    img_save_path = os.path.join(save_dir, os.path.basename(image_path))
    cv2.imwrite(img_save_path, img)
    print(f"Saved: {img_save_path} and {label_path}")

def semi_supervised_labeling(image_dir, save_dir, min_area=250):
    """
    For each image, show bounding boxes and ask user to accept or skip.
    If accepted, save YOLO label and image.
    """
    os.makedirs(save_dir, exist_ok=True)
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for image_name in images:
        image_path = os.path.join(image_dir, image_name)
        closingImage, boundRect = adaptive_threshold_and_filter(image_path, min_area=min_area)
        if not boundRect:
            print(f"No bounding boxes found for {image_name}. Skipping.")
            continue
        # Condition on number of bounding boxes
        if len(boundRect) != 4 and len(boundRect) != 11:
            print(f"Too many bounding boxes ({len(boundRect)}) found in {image_name}. Skipping.")
            continue
        # Print the number of bounding boxes found
        print(f"Found {len(boundRect)} bounding boxes in {image_name}.")
        # Show the bounding boxes (already done in adaptive_threshold_and_filter)
        resp = input(f"Accept bounding boxes for {image_name}? (y/n): ").strip().lower()
        if resp == "y":
            save_yolo_labels(image_path, boundRect, save_dir)
        else:
            print("Skipped.")

# Example usage:
#semi_supervised_labeling("cropped_images", "cropped_yolo_dataset")


# Croping sealed images and saving them in a new folder

import os
import cv2
import matplotlib.pyplot as plt

def extract_and_save_seal_bboxe_image(labeled_samples_dir="labeled_samples", label_text_file ="../yolo_label_sealed"  ,seal_output_dir="yolo_label_sealed"):
    """
    For each labeled image in labeled_samples, extract bounding boxes with class 'sealed' (or your seal class id)
    and save the cropped seal image and its YOLO label in a new folder.
    """
    # Set your seal class id (e.g., 1 if 'sealed' is class 1 in your data.yaml)
    SEAL_CLASS_ID = 1

    # os.makedirs(seal_output_dir, exist_ok=True)

    for img_id in os.listdir(labeled_samples_dir):
        id_folder = os.path.join(labeled_samples_dir, img_id)
        if not os.path.isdir(id_folder):
            continue
        for image in os.listdir(id_folder):
            if not image.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            image_no_ext = os.path.splitext(image)[0]
            label_path = os.path.join(label_text_file, image_no_ext + '.txt')
            image_path = os.path.join(id_folder, image)
            if not os.path.exists(label_path):
                continue
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image {image_path}")
                continue
            h, w = img.shape[:2]
            with open(label_path, 'r') as f:
                lines = f.readlines()
                yolo_label = ""
                for idx, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    class_id = int(parts[0])
                    if class_id == 0:
                        class_name = 'unsealed'
                    elif class_id == SEAL_CLASS_ID:
                        class_name = 'sealed'
                    # Get bounding box coordinates of yolo format
                    x_min = int(float(parts[1]) * w - float(parts[3]) * w / 2)
                    y_min = int(float(parts[2]) * h - float(parts[4]) * h / 2)
                    x_max = int(float(parts[1]) * w + float(parts[3]) * w / 2)
                    y_max = int(float(parts[2]) * h + float(parts[4]) * h / 2)
                    
                    # Crop and save seal image
                    if class_id == SEAL_CLASS_ID:
                        os.makedirs(seal_output_dir, exist_ok=True)
                        cropped_img = img[y_min:y_max, x_min:x_max]
                        crop_name = f"{image_no_ext}_seal_{idx}.jpg"
                        crop_path = os.path.join(seal_output_dir, crop_name)
                        # cv2.imwrite(crop_path, cropped_img)
                        print(f"Saved: {crop_path}")
                    
                    # Save YOLO label for this crop
                    x_center = ((x_min + x_max) / 2) / w
                    y_center = ((y_min + y_max) / 2) / h
                    bbox_w = (x_max - x_min) / w
                    bbox_h = (y_max - y_min) / h
                    yolo_label += f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}\n"
                    
                    # Draw rectangle on the image with the class id with different color
                    color = (0, 255, 0) if class_id == SEAL_CLASS_ID else (255, 0, 0)
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
                    cv2.putText(img, f"{class_name}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                
                # Save the labeled image with bounding box
                cv2.imwrite(image_path, img)
                
                # Display the bounding box on the image
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.title(f"Image: {image_no_ext}, Class: {class_name}")
                plt.show()
                
                # Save the YOLO label file
                label_save_path = os.path.join(id_folder, f"{image_no_ext}_seal.txt")
                with open(label_save_path, "w") as label_file:
                    label_file.write(yolo_label)
                print(f"Saved: {label_save_path}")

# Example usage:
#extract_and_save_seal_bboxes_image(labeled_samples_dir="labeled_samples", label_text_file ="../yolo_label_sealed"  ,seal_output_dir="yolo_label_sealed")