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
        
        model_path_seal = "runs/detect/train_seal_detection2/" + model_path
        if not os.path.exists(model_path_seal):
            raise FileNotFoundError(f"Model file not found: {model_path_seal}")
        else:
            model_seal = YOLO(model_path_seal)
            results_seal = model_seal(image_path, conf=conf, iou=iou)
            results_type.append(results_seal)
    
    elif 'code' in object_type:
        print("Detecting code...")
        
        model_path_code = "runs/detect/train_code_detection2/" + model_path
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
    
    
# Model CNN character recognition
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# Dataset
class CharDataset(Dataset):
    def __init__(self, folder):
        self.data = []
        self.labels = []
        self.classes = sorted(list(set([f.split("_")[0] for f in os.listdir(folder)])))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.folder = folder
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        for fname in os.listdir(folder):
            label = fname.split("_")[0]
            self.data.append(fname)
            self.labels.append(self.class_to_idx[label])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder, self.data[idx])
        image = self.transform(Image.open(img_path))
        label = self.labels[idx]
        return image, label

# Model
class CharCNN(nn.Module):
    def __init__(self, num_classes):
        super(CharCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Load the model for inference
def load_model(model_path, num_classes):
    model = CharCNN(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

class AdvancedCharCNN(nn.Module):
    def __init__(self, num_classes):
        super(AdvancedCharCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load the model for inference
def load_model(model_path, num_classes):
    model = AdvancedCharCNN(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_characters(model, image_paths, transform):
    model.eval()
    predictions = []
    with torch.no_grad():
        for img_path in image_paths:
            image = transform(Image.open(img_path)).unsqueeze(0)  # Add batch dimension
            output = model(image)
            pred = output.argmax(dim=1).item()
            predictions.append(pred)
            text = text + AdvancedCharCNN('augmented_characters').classes[pred]
    return text, predictions

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

def predict_characters(model, image_paths, transform, class_names):
    model.eval()
    predictions = []
    text = ""
    with torch.no_grad():
        for img_path in image_paths:
            image = transform(Image.open(img_path)).unsqueeze(0)
            output = model(image)
            pred = output.argmax(dim=1).item()
            predictions.append(pred)
            text += class_names[pred]
    return text, predictions

# Example usage
if __name__ == "__main__":
    # Training
    dataset = CharDataset("augmented_characters")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = CharCNN(num_classes=len(dataset.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        total_loss = 0
        for x, y in loader:
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}: loss = {total_loss:.4f}")
        
    # Save the model
    torch.save(model.state_dict(), "char_cnn.pth")
    # Inference 31 classes
    model = load_model("char_cnn.pth", num_classes=len(CharDataset("augmented_characters").classes))
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    print("Model classes:", CharDataset("augmented_characters").classes)
    
    # Replace with your image paths
    test_image_paths = ["cropped_characters\G_labeled_1-132130001-OCR-LF-C01.jpg_-1.jpg", "cropped_characters\7_labeled_1-143242001-OCR-LB-C02.jpg_0.jpg"]
    predictions = predict_characters(model, test_image_paths, transform)[1]
    
    for img_path, pred in zip(test_image_paths, predictions):
        print(f"Image: {img_path}, Predicted Class: {CharDataset('augmented_characters').classes[pred]}")