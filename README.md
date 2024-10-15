# Simple Number Plate Recognition

This project performs automatic number plate recognition (ANPR) using OpenCV, EasyOCR, and other Python libraries. The program detects a license plate in an image, processes it, and extracts the text using Optical Character Recognition (OCR).

## Features
- License plate detection using OpenCV and image processing techniques.
- Text recognition from license plates using EasyOCR.
- Edge detection, contour finding, and masking for accurate plate localization.
- Visualization of processing steps with `matplotlib`.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/simple-number-plate-recognition.git
```
2. Navigate to the project directory:

```bash
cd simple-number-plate-recognition
```
3. Install the dependencies:

```bash
pip install -r requirements.txt
```

### Usage

To run the project, use the following command:

```bash
python app.py
```

Make sure to place your images in the plat_nomor_dataset folder.

### File Structure
```bash
Copy code
simple_number_plate_recognition/
│
├── anpr.py                  # Main script for license plate recognition
├── plat_nomor_dataset/       # Folder containing number plate images
├── requirements.txt          # Required Python packages
└── README.md                 # Project documentation
```