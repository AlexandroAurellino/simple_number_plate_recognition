import cv2
import numpy as np
import imutils
import easyocr
import os
import matplotlib.pyplot as plt

def load_image(file_path):
    """Load an image from the specified path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Unable to read the image file: {file_path}")
    print(f"Image loaded successfully. Shape: {img.shape}")
    return img

def preprocess_image(img):
    """Apply grayscale and bilateral filter to the image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    print(f"Preprocessing complete. Shape after preprocessing: {filtered.shape}")
    return filtered

def detect_edges(img):
    """Apply Canny edge detection to the image."""
    edged = cv2.Canny(img, 30, 200)  # Increased upper threshold
    print(f"Edge detection complete. Shape after edge detection: {edged.shape}")
    return edged

def find_license_plate_contour(img):
    """Find the contour of the license plate in the image."""
    keypoints = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)  # Reduced epsilon value
        if len(approx) == 4:
            location = approx
            break
    
    if location is None:
        print("No license plate contour found.")
    else:
        print(f"License plate contour found. Shape: {location.shape}")
    return location

def create_mask(img, location):
    """Create a mask for the license plate area."""
    mask = np.zeros(img.shape[:2], np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    masked = cv2.bitwise_and(img, img, mask=mask)
    print(f"Mask created. Shape of masked image: {masked.shape}")
    print(f"Number of non-zero pixels in mask: {np.count_nonzero(mask)}")
    return masked, mask

def crop_license_plate(img, mask):
    """Crop the license plate area from the image."""
    (x, y) = np.where(mask == 255)
    if len(x) == 0 or len(y) == 0:
        raise ValueError("No non-zero pixels found in the mask.")
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped = img[x1:x2+1, y1:y2+1]
    print(f"License plate cropped. Shape of cropped image: {cropped.shape}")
    return cropped

def read_license_plate(img):
    """Use EasyOCR to read the text on the license plate."""
    reader = easyocr.Reader(['en'])
    result = reader.readtext(img)
    print(f"OCR result: {result}")
    return result

def draw_result(img, text, location):
    """Draw the recognized text and bounding box on the original image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    res = cv2.putText(img, text=text, org=(location[0][0][0], location[1][0][1]+60),
                    fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
    res = cv2.rectangle(img, tuple(location[0][0]), tuple(location[2][0]), (0,255,0), 3)
    return res

def visualize_steps(original, gray, edged, masked, cropped):
    """Visualize the steps of the license plate recognition process."""
    plt.figure(figsize=(20, 10))
    images = [original, gray, edged, masked, cropped]
    titles = ['Original', 'Grayscale', 'Edged', 'Masked', 'Cropped']
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(2, 3, i+1)
        if len(img.shape) == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def recognize_license_plate(file_path):
    """Main function to recognize license plate."""
    try:
        # Load image
        img = load_image(file_path)
        
        # Preprocess image
        processed_img = preprocess_image(img)
        
        # Detect edges
        edged = detect_edges(processed_img)
        
        # Find license plate contour
        location = find_license_plate_contour(edged)
        
        if location is None:
            print("No license plate found in the image.")
            return None
        
        # Create mask and crop license plate
        masked, mask = create_mask(img, location)
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        license_plate = crop_license_plate(gray, mask)
        
        # Visualize steps
        visualize_steps(img, processed_img, edged, masked, license_plate)
        
        # Read license plate
        result = read_license_plate(license_plate)
        
        if not result:
            print("No text detected on the license plate.")
            return None
        
        # Draw result
        text = result[0][-2]
        result_img = draw_result(img, text, location)
        
        # Display result
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.show()
        
        return text
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    return None

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "..", "plat_nomor_dataset")
    file_name = "AB3226FR.jpg"  # Using the file name from your last run
    file_path = os.path.join(dataset_path, file_name)
    
    print(f"Attempting to process image: {file_path}")
    recognized_text = recognize_license_plate(file_path)
    if recognized_text:
        print(f"Recognized license plate: {recognized_text}")
    else:
        print("Failed to recognize license plate.")