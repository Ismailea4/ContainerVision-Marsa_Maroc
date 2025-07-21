import argparse
from src.pipeline import container_OCR
import cv2
import os

def main():
    parser = argparse.ArgumentParser(description="ContainerVision - Marsa Maroc OCR & Seal Detection")
    parser.add_argument('--image', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--model', type=str, default='weights/best.pt', help='Path to YOLO model weights')
    parser.add_argument('--char_model', type=str, default='char_cnn.pth', help='Path to character CNN model')
    parser.add_argument('--object_type', type=str, nargs='+', default=['seal', 'code', 'character'], help='Object types to detect')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--display', action='store_true', help='Display images with predictions')
    parser.add_argument('--output', type=str, default='output_with_predictions.jpg', help='Path to save output image')

    args = parser.parse_args()
    
    # check if the image path is a directory or a file
    if os.path.isdir(args.image):
        print(f"Processing all images in directory: {args.image}")
        image_paths = [os.path.join(args.image, f) for f in os.listdir(args.image) if f.endswith(('.jpg', '.png', '.jpeg'))]
        results = []
        for image_path in image_paths:
            print(f"Processing image: {image_path}")
            result = container_OCR(
                image_path=image_path,
                model_path=args.model,
                object_type=args.object_type,
                conf=args.conf,
                iou=args.iou,
                display=args.display
            )
            results.append(result)
            print("Extracted Codes:", result['code'])
    
    else:
        print(f"Processing single image: {args.image}")
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Image file not found: {args.image}")
        
        image_path = args.image

        result = container_OCR(
            image_path=args.image,
            model_path=args.model,
            object_type=args.object_type,
            conf=args.conf,
            iou=args.iou,
            display=args.display
        )

        print("Extracted Codes:", result['code'])

        # Save the final image with predictions
        cv2.imwrite(args.output, result['predictions'])
        print(f"Output image saved to {args.output}")

if __name__ == "__main__":
    main()