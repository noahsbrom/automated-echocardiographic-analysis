import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

model_path = 'model.h5'
test_image_path = '_2023-05-31-16-26-00.tif'

# Load the saved model
model = tf.keras.models.load_model(model_path)

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

def calculate_line_length_mm(x_start, y_start, x_end, y_end, real_height_mm=10.5, image_height_pixels=512):
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

def process_image_with_model(model, image_path):
    preprocessed_img = crop_and_preprocess_image(image_path)
    if preprocessed_img is not None:
        draw_prediction_on_image(model, preprocessed_img, image_path)

process_image_with_model(model, test_image_path)
