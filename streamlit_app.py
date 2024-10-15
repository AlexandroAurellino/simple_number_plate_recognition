import os
import cv2
import numpy as np
import imutils
import easyocr
import streamlit as st
from PIL import Image
import io
import base64

# Function to check if the file has an allowed extension
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess the image and recognize license plate
def recognize_license_plate(image, d, sigma_color, sigma_space, lower_threshold, upper_threshold):
    # Convert PIL Image to OpenCV format
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Preprocessing steps
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, d, sigma_color, sigma_space)
    edged = cv2.Canny(filtered, lower_threshold, upper_threshold)
    
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

    # Create mask and crop the license plate
    if location is not None:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [location], 0, 255, -1)
        masked_img = cv2.bitwise_and(img, img, mask=mask)

        x, y = np.where(mask == 255)
        cropped = img[np.min(x):np.max(x)+1, np.min(y):np.max(y)+1]

        # Use EasyOCR to detect text from the cropped image
        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped)

        # Extract text if available
        if result:
            return result[0][-2], filtered, edged, masked_img, cropped
    
    return None, filtered, edged, None, None

# Convert OpenCV image to base64 for displaying
def cv2_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

# Streamlit app
def main():
    st.title("License Plate Recognition App")

    # Sidebar for parameter tuning
    st.sidebar.header("Parameter Tuning")
    
    d = st.sidebar.slider("Bilateral Filter: Diameter", 5, 100, 70)
    sigma_color = st.sidebar.slider("Bilateral Filter: Sigma Color", 10, 200, 70)
    sigma_space = st.sidebar.slider("Bilateral Filter: Sigma Space", 10, 200, 70)
    lower_threshold = st.sidebar.slider("Canny Edge: Lower Threshold", 0, 255, 10)
    upper_threshold = st.sidebar.slider("Canny Edge: Upper Threshold", 0, 255, 100)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        if allowed_file(uploaded_file.name):
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            if st.button('Recognize License Plate'):
                text, filtered, edged, masked_img, cropped = recognize_license_plate(
                    image, d, sigma_color, sigma_space, lower_threshold, upper_threshold
                )

                if text is not None:
                    st.success(f"Recognized Text: {text}")
                    
                    # Display intermediate steps using columns
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.subheader("Bilateral Filter")
                        st.image(filtered, caption='Bilateral Filter', use_column_width=True)
                    
                    with col2:
                        st.subheader("Edge Detection")
                        st.image(edged, caption='Edge Detection', use_column_width=True)
                    
                    with col3:
                        st.subheader("Masked Image")
                        st.image(masked_img, caption='Masked Image', use_column_width=True)
                    
                    with col4:
                        st.subheader("Cropped License Plate")
                        st.image(cropped, caption='Cropped License Plate', use_column_width=True)
                else:
                    st.error("License plate could not be recognized.")
                    # Display intermediate steps even if recognition fails
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Bilateral Filter")
                        st.image(filtered, caption='Bilateral Filter', use_column_width=True)
                    with col2:
                        st.subheader("Edge Detection")
                        st.image(edged, caption='Edge Detection', use_column_width=True)
        else:
            st.error("Invalid file type. Please upload a JPG, JPEG, or PNG image.")

if __name__ == "__main__":
    main()