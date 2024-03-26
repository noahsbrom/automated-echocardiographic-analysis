import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model_path = 'test.h5'
test_image_path = '_2022-12-12-11-05-41.png'

# Load the saved model
model = tf.keras.models.load_model(model_path)

# Function to draw a line on an image using model predictions
def draw_prediction_on_image(model, image_path):

    # Load and preprocess the image
    img = load_img(image_path, target_size=(512, 512))
    img_array = img_to_array(img) / 255.0  # Normalize to 0-1 range
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the coordinates
    predicted_coords = model.predict(img_array)[0] * 512  # Rescale back from normalization
    x, y_start, y_end = [int(coord) for coord in predicted_coords]  # Convert to integer for drawing

    # Draw the line on the image
    drawn_img = cv2.imread(image_path)  # Read in the original image
    drawn_img = cv2.resize(drawn_img, (512, 512))  # Resize to match model input
    cv2.line(drawn_img, (x, y_start), (x, y_end), (255, 0, 0), 5)  # Draw line

    # Display the image
    cv2.imshow('Predicted Line', drawn_img)
    cv2.waitKey(0)  # Wait for a key press to close the image window
    cv2.destroyAllWindows()

    # Save the image with the line drawn
    output_path = os.path.join(os.path.dirname(image_path), 'predicted_' + os.path.basename(image_path))
    cv2.imwrite(output_path, drawn_img)
    print(f"Saved annotated image to {output_path}")

# Run the function with model and image
draw_prediction_on_image(model, test_image_path)
