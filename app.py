from flask import Flask, request, render_template, send_from_directory, url_for, Response, make_response, send_file
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import tensorflow as tf
import pytesseract
from pytesseract import Output
import re
import csv
import zipfile
from io import BytesIO, StringIO


app = Flask(__name__)
model_path = 'model.h5'
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Load model here
model = tf.keras.models.load_model(model_path)

# Global list to store results
batch_results = []
processed_files = []

def add_batch_result(result):
    """Adds a new result to the batch results."""
    global batch_results
    batch_results.append(result)

def retrieve_batch_results():
    """Retrieves all batch results stored in memory."""
    global batch_results
    return batch_results

# Define your functions here: crop_and_preprocess_image, calculate_line_length_mm,
# draw_prediction_on_image, read_study_series_text, calculate_scale_height, etc.

def crop_and_preprocess_image(image_path):
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None

    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = image[y:y+h, x:x+w]
        resized_image = cv2.resize(cropped_image, (512, 512), interpolation=cv2.INTER_AREA)
        return resized_image
    else:
        print("No contours found in image.")
        return None

def calculate_line_length_mm(x_start, y_start, x_end, y_end, real_height_mm=12.5, image_height_pixels=512):
    scale_pixels_per_mm = image_height_pixels / real_height_mm
    line_length_pixels = np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
    line_length_mm = line_length_pixels / scale_pixels_per_mm
    return line_length_mm

def draw_prediction_on_image(model, preprocessed_img_array, original_image_path):
    img_array = np.expand_dims(preprocessed_img_array / 255.0, axis=0)
    predicted_coords = model.predict(img_array)[0] * 512
    x, y_start, y_end = [int(coord) for coord in predicted_coords]

    cv2.line(preprocessed_img_array, (x, y_start), (x, y_end), (255, 0, 0), 5)

    line_length_mm = calculate_line_length_mm(x, y_start, x, y_end)
    annotation_text = f"{line_length_mm:.2f} mm"
    cv2.putText(preprocessed_img_array, annotation_text, (x, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2)

    output_path = os.path.join(os.path.dirname(original_image_path), 'predicted_' + os.path.basename(original_image_path))
    cv2.imwrite(output_path, preprocessed_img_array)
    print(f"Saved annotated image to {output_path}")

def crop_scale_area(image_path):
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None

    # Check if the image has an alpha channel; if yes, convert to BGR
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary threshold to create a binary image for contour detection
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Extra margins
        extra_height = int(h * 0.02)
        extra_width = int(w * 0.1)
        right_extension = int(w * 0.01)

        y_start = max(0, y - extra_height)
        y_end = min(image.shape[0], y + h + extra_height)
        x_end = min(image.shape[1], x + w + extra_width + right_extension)

        # Crop the region of interest
        scale_area = gray[y_start:y_end, x+w:x_end]  # Ensure to use 'gray' if further processing needs grayscale

        return scale_area
    else:
        print("No contours found in image.")
        return None

def calculate_scale_height_function_1(image):
    # Check if image is None (not loaded properly)
    # Inverting the image colors for better OCR accuracy
    inverted_image = cv2.bitwise_not(image)

    # Apply OCR to the inverted image to detect numbers
    text_detected = pytesseract.image_to_string(inverted_image, config='--psm 6 outputbase digits')

    # Use regular expression to find all numbers in the detected text
    numbers_detected = re.findall(r"[-+]?\d*\.\d+|\d+", text_detected)

    # Convert the extracted strings to float
    numbers_detected = [float(num) for num in numbers_detected]

    # Calculate the height of the scale if two or more numbers are detected
    if len(numbers_detected) >= 2:
        # Sort the numbers to ensure the correct calculation
        numbers_detected.sort()
        # Assuming the first and last numbers correspond to the top and bottom of the scale
        scale_height = numbers_detected[-1] - numbers_detected[0]
    else:
        return "Not enough numbers detected."

    # Return the calculated scale height and the detected numbers
    return scale_height, numbers_detected

def calculate_scale_height_function_2(image):
    # Apply binary threshold to the image to enhance the numbers
    _, thresholded_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

    # Invert the colors for better OCR accuracy
    inverted_image = cv2.bitwise_not(thresholded_image)

    # Apply OCR to the inverted image to detect numbers
    custom_config = r'--psm 6 outputbase digits -c tessedit_char_whitelist=0123456789.'
    text_detected = pytesseract.image_to_string(inverted_image, config=custom_config)

    # Use regular expression to find all numbers in the detected text
    numbers_detected = re.findall(r"[-+]?\d*\.\d+|\d+", text_detected)

    # Truncate numbers to one decimal place and convert to float
    numbers_detected = [float(f'{float(num):.1f}') for num in numbers_detected]

    # Sort the numbers to ensure the correct calculation
    numbers_detected.sort()

    # Calculate the height of the scale if two or more numbers are detected
    if len(numbers_detected) >= 2:
        # Assuming the first and last numbers correspond to the top and bottom of the scale
        scale_height = numbers_detected[-1] - numbers_detected[0]
    else:
        scale_height = "Not enough numbers detected."

    # Return the calculated scale height and the detected numbers
    return scale_height, numbers_detected

def read_study_series_text(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Ensure the image was loaded successfully
    if image is None:
        return "Image not loaded", "Image not loaded"

    h, w, _ = image.shape
    cropped_image = image[:int(h * 0.2), :int(w * 0.4)]  # Adjust this if necessary

    # Convert the cropped image to grayscale
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to get a binary image
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert the binary image to get white background and black letters
    inverted_image = 255 - binary_image

    # Use pytesseract to extract text with optimized psm setting
    custom_config = r'--oem 3 --psm 11'
    text_data = pytesseract.image_to_string(inverted_image, config=custom_config, output_type=Output.DICT)
    text = text_data['text']

    # Define keywords to ignore, adding 'vevo' and '3100'
    ignore_keywords = {"study", "series", "frequency", "vevo", "3100"}

    # Using regex to find the content after "Study" until "Series", and from "Series" until "Frequency"
    study_regex = r"(?<=Study)(.*?)(?=Series|$)"
    series_regex = r"(?<=Series)(.*?)(?=Frequency|$)"

    # Finding matches
    study_match = re.search(study_regex, text, re.IGNORECASE | re.DOTALL)
    series_match = re.search(series_regex, text, re.IGNORECASE | re.DOTALL)

    # Function to process text by removing unwanted words and stripping extra whitespace
    def process_text(text):
        if not text:
            return "Not found"
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in ignore_keywords]
        return ' '.join(filtered_words).strip()

    # Process the extracted text
    study_text = process_text(study_match.group(0) if study_match else "")
    series_text = process_text(series_match.group(0) if series_match else "")

    return study_text, series_text

def calculate_scale_height(image_path):
    scale_area = crop_scale_area(image_path)
    if scale_area is None:
        return None, "Crop function failed."

    scale_height, numbers_detected = calculate_scale_height_function_1(scale_area)
    print(f"Detected numbers f1: {numbers_detected}")
    print(f"Scale height f1: {scale_height}")
    if scale_height is None or scale_height < 8.5 or scale_height > 14 or has_more_than_one_decimal(scale_height):
        scale_height, numbers_detected = calculate_scale_height_function_2(scale_area)

    return scale_height, numbers_detected

def has_more_than_one_decimal(number):
    """
    Check if the number has more than one decimal place of precision.
    """
    if number is not None:
        # Convert number to string and split by decimal point
        number_str = f"{number:.10f}".rstrip('0')  # Format to string and remove trailing zeros
        if '.' in number_str:
            decimal_part = number_str.split('.')[1]
            return len(decimal_part) > 1  # More than one decimal place
    return False




@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Clear previous batch results
        global processed_files
        processed_files = []

        files = request.files.getlist('file')
        results = []
        for file in files:
            if file:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                study_info, series_info = read_study_series_text(filepath)
                processed_filename, scale_height, line_length = process_image_with_model(model, filepath, app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER'])
                if processed_filename:
                    result = (processed_filename, study_info, series_info, scale_height, line_length)
                    results.append(result)
                    add_batch_result(result)  # Add result to global storage

        return render_template('batch_view.html', results=results)
    return render_template('upload.html')


# Define the process_image_with_model function and all necessary utility functions

def process_image_with_model(model, image_path, upload_folder, processed_folder):
    global processed_files
    original_filename = os.path.basename(image_path)
    original_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Read the original image

    # Check if the image was loaded successfully
    if original_image is None:
        print(f"Error: Could not load image {image_path}")
        return None, "Image not loaded", None

    # Calculate scale height from the original image
    scale_height, scale_detection_status = calculate_scale_height(image_path)

    # Proceed only if the scale_height is successfully calculated
    if scale_height is None:
        print(f"Scale height calculation failed: {scale_detection_status}")
        return None, scale_detection_status, None

    # Preprocess the image for model prediction
    preprocessed_img = crop_and_preprocess_image(image_path)
    if preprocessed_img is not None:
        output_filename = 'predicted_' + original_filename
        output_path = os.path.join(processed_folder, output_filename)

        # Use the preprocessed image for model predictions
        img_array = np.expand_dims(preprocessed_img / 255.0, axis=0)
        predicted_coords = model.predict(img_array)[0] * 512
        x, y_start, y_end = [int(coord) for coord in predicted_coords]

        # Calculate line length using the scale height as real height
        line_length_mm = calculate_line_length_mm(x, y_start, x, y_end, scale_height, preprocessed_img.shape[0])
        line_length_mm = round(line_length_mm, 2)  # Round line length to two decimal places

        # Annotate the processed image with the prediction line and scale height
        cv2.line(preprocessed_img, (x, y_start), (x, y_end), (255, 0, 0), 5)
        annotation_text = f"Left Atrium Length: {line_length_mm} mm"
        cv2.putText(preprocessed_img, annotation_text, (10, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Save the annotated image
        cv2.imwrite(output_path, preprocessed_img)
        print(f"Saved annotated image to {output_path}")

        # Add this file to the processed_files list
        processed_files.append(output_filename)

        return output_filename, f"{scale_height:.2f} mm", line_length_mm
    else:
        print("Image preprocessing failed.")
        return None, "Image preprocessing failed", None



@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/export_data', methods=['POST'])
def export_data():
    study = request.form['study']
    series = request.form['series']
    line_length = request.form['line_length']

    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(['Study', 'Series', 'Line Length'])
    cw.writerow([study, series, line_length])

    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=export.csv"
    output.headers["Content-type"] = "text/csv"
    return output

@app.route('/download_all_images')
def download_all_images():
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        for file_name in processed_files:
            full_path = os.path.join(app.config['PROCESSED_FOLDER'], file_name)
            zf.write(full_path, arcname=file_name)
    memory_file.seek(0)
    return send_file(memory_file, download_name='processed_images.zip', as_attachment=True, mimetype='application/zip')


@app.route('/export_all_data', methods=['GET'])
def export_all_data():
    results = retrieve_batch_results()  # Use the retrieve function to get all results

    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(['Study', 'Series', 'Line Length'])

    for result in results:
        study, series, line_length = result[1], result[2], result[4]
        cw.writerow([study, series, line_length])

    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=all_data.csv"
    output.headers["Content-type"] = "text/csv"
    return output

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

if __name__ == '__main__':
    app.run(debug=True)
