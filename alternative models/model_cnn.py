import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model

# Directory containing images
image_dir = 'testoutput'
# Suffix for coordinate files
coord_suffix = '_coordinates.txt'

def load_dataset(image_dir):
    images = []
    labels = []

    print("Loading dataset...")
    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):
            img_path = os.path.join(image_dir, filename)
            img = load_img(img_path, target_size=(512, 512))  # Make sure the size matches your input size
            img = img_to_array(img)
            img /= 255.0  # Normalize image pixels to [0,1]
            images.append(img)

            base_name = os.path.splitext(filename)[0]
            coord_file = os.path.join(image_dir, base_name + coord_suffix)
            with open(coord_file, 'r') as f:
                coords = [float(num)/512 for num in f.read().split(',')]  # Adjust normalization if necessary
                labels.append(coords)

    print(f"Loaded {len(images)} images and {len(labels)} labels.")
    return np.array(images), np.array(labels)

def create_custom_cnn(input_shape):
    inputs = Input(shape=input_shape)
    # Custom CNN architecture
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(3, activation='linear')(x)  # Assuming 3 coordinates for each label
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Load your dataset
images, labels = load_dataset(image_dir)

# Split into training and testing sets
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
print(f"Training on {len(X_train)} samples, validating on {len(X_test)} samples.")

# Create the model
input_shape = (512, 512, 3)  # Change according to your image size
print("Creating custom model...")
model = create_custom_cnn(input_shape)

# Compile the model
print("Compiling model...")
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
print("Training model...")
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))  # Adjust epochs as needed

# Evaluate the model
print("Evaluating model...")
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test loss: {test_loss}, Test accuracy: {test_acc}')

# Predict on the test set
predicted_coords = model.predict(X_test)

# Compare the actual and predicted coordinates for the first few test images
num_examples_to_show = 5
for i in range(num_examples_to_show):
    print(f"Image {i}:")
    print(f"Actual coordinates: {y_test[i] * 512}")  # Rescale back from normalization
    print(f"Predicted coordinates: {predicted_coords[i] * 512}")  # Rescale back
