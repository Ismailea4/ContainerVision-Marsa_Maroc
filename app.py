from flask import Flask, request, jsonify, send_file, render_template, Response
import os
import cv2
import numpy as np
from src.pipeline import container_OCR, container_detection
from werkzeug.utils import secure_filename
from io import BytesIO
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Run your OCR pipeline
    print("the file ame is ",filepath)
    result = container_detection(filepath, object_type = ['code','seal'], display=False)
    code = result['detections']
    image = result['predictions']  # This is a numpy array (OpenCV BGR)
    
    print('the code is',code)

    # Convert image to PNG in memory
    _, img_encoded = cv2.imencode('.png', image)
    img_bytes = BytesIO(img_encoded.tobytes())

    # Create a response and set custom headers
    response = send_file(
        img_bytes,
        mimetype='image/png',
        as_attachment=False,
        download_name='result.png'
    )
    # Add custom header for code
    response.headers['codes'] = str(code)
    return response

@app.route('/detect_json', methods=['POST'])
def detect_json():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    result = container_OCR(filepath, display=False)
    code = result['code']
    # Optionally, you can encode the image as base64 if you want to return it in JSON

    return jsonify({'code': code})

if __name__ == '__main__':
    app.run(debug=True)