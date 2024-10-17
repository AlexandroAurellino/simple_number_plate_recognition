import os
import cv2
import numpy as np
import imutils
import easyocr
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Initialize Flask app
app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess the image and recognize license plate
def recognize_license_plate(image_path):
    print(f"Processing image at: {image_path}")
    img = cv2.imread(image_path)

    if img is None:
        print("Error loading image")
        return None, None

    # Preprocessing steps
    try:
        print("Preprocessing image...")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("Image converted to grayscale")
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        print("Image bilateral filtered")
        edged = cv2.Canny(filtered, 30, 200)
        print("Image edges detected")

        # Find contours
        contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break

        if location is None:
            print("No license plate contour found")
            return None, None

        # Create mask and crop the license plate
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [location], 0, 255, -1)
        masked_img = cv2.bitwise_and(img, img, mask=mask)

        x, y = np.where(mask == 255)
        cropped = img[np.min(x):np.max(x)+1, np.min(y):np.max(y)+1]

        # Use EasyOCR to detect text from the cropped image
        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped)
        print(f"OCR result: {result}")

        # Extract text if available
        if result:
            return result[0][-2], cropped
        else:
            return None, None

    except Exception as e:
        print(f"Error in processing: {e}")
        return None, None


# Convert image to base64 for displaying on the webpage
def image_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    io_buf = BytesIO(buffer)
    processed_img_data = base64.b64encode(io_buf.getvalue()).decode('utf-8')
    print(f"Processed image data (Base64): {processed_img_data[:100]}...")  # Only print the first 100 characters
    return base64.b64encode(io_buf.getvalue()).decode('utf-8')

# Home route to handle the file upload
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        text, processed_image = recognize_license_plate(file_path)

        if text is not None and processed_image is not None:
            print(f"Recognized Text: {text}")
            processed_img_data = image_to_base64(processed_image)
            return render_template('result.html', text=text, image_data=processed_img_data)
        else:
            print("No license plate detected.")
            return "License plate could not be recognized.", 400

    return render_template('index.html')


# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Run the app
if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, host="0.0.0.0")


    
