# Loading data for machine learning model
# Created for LA images
# Martinus Kleiweg

import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf


import tensorflow as tf

def conv_block(inputs, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x

def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = tf.keras.layers.MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip_features, num_filters):
    x = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = tf.keras.layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def unet_model(input_shape):
    inputs = tf.keras.Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = tf.keras.Model(inputs, outputs, name="U-Net")
    return model

model = unet_model((256, 256, 1))  # Adjust input shape to match your dataset
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


def load_image_paths(directory):
    file_paths = []
    filenames = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            file_path = os.path.join(directory, filename)
            file_paths.append(file_path)
            filenames.append(filename)
    return file_paths, filenames

def save_processed_images(file_paths, save_directory, reduce_factor=None, crop_box=None):
    """
    Processes and saves images from file paths. The processing includes conversion to grayscale,
    optional resizing, and optional cropping.

    Parameters:
        file_paths (list): List of file paths to process.
        save_directory (str): Directory to save processed images.
        reduce_factor (float, optional): Factor to scale down images. If None, no resizing is applied.
        crop_box (tuple, optional): The crop box is a 4-tuple defining the left, upper, right, and lower pixel coordinate.
    """
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    for file_path in tqdm(file_paths, desc="Processing and saving images"):
        with Image.open(file_path) as img:
            # Convert the image to grayscale
            img = img.convert('L')
            # Crop the image if a crop box is provided
            if crop_box:
                img = img.crop(crop_box)
            # Resize the image if a reduce factor is provided
            if reduce_factor:
                original_size = img.size
                new_size = (int(original_size[0] * reduce_factor), int(original_size[1] * reduce_factor))
                img = img.resize(new_size, Image.ANTIALIAS)
            # Save the processed image
            base_name = os.path.basename(file_path)
            new_filename = os.path.splitext(base_name)[0] + '.png'
            save_path = os.path.join(save_directory, new_filename)
            img.save(save_path, 'PNG')

def batch_load_saved_images(directory, batch_size=10):
    filenames = sorted([f for f in os.listdir(directory) if f.endswith('.png')])
    for start_idx in tqdm(range(0, len(filenames), batch_size), desc="Loading images in batches"):
        batch_images = []
        for idx in range(start_idx, min(start_idx + batch_size, len(filenames))):
            file_path = os.path.join(directory, filenames[idx])
            with Image.open(file_path) as img:
                numpy_image = np.array(img)
                batch_images.append(numpy_image.astype('float32') / 255.0)
        yield np.array(batch_images)

def check_and_process_images(input_directory, output_directory, processed_dirs, reduce_factor=None, test_size=0.2, val_size=0.25):
    # Load all image paths
    input_paths, input_filenames = load_image_paths(input_directory)
    output_paths, output_filenames = load_image_paths(output_directory)
    assert input_filenames == output_filenames, "Mismatch between input and output filenames"

    # Split data into training, validation, and test sets
    (train_input, train_output), (val_input, val_output), (test_input, test_output) = split_data(input_paths, output_paths, test_size, val_size)

    # Initialize dictionaries to store batch loaders for each dataset
    batch_loaders = {}

    # Process and batch-load images for each set
    for set_type, (input_set, output_set) in zip(['train', 'val', 'test'], [(train_input, train_output), (val_input, val_output), (test_input, test_output)]):
        processed_input_directory, processed_output_directory = processed_dirs[set_type]

        if not os.path.exists(processed_input_directory) or not os.listdir(processed_input_directory) or not os.path.exists(processed_output_directory) or not os.listdir(processed_output_directory):
            print(f"Processing and saving {set_type} images...")
            save_processed_images(input_set, processed_input_directory, reduce_factor=reduce_factor)
            save_processed_images(output_set, processed_output_directory, reduce_factor=reduce_factor)

        batch_loaders[set_type] = (
            batch_load_saved_images(processed_input_directory),
            batch_load_saved_images(processed_output_directory)
        )

    return batch_loaders


def split_data(input_paths, output_paths, test_size=0.2, val_size=0.25):
    """
    Splits the data into training, validation, and test sets.
    test_size: Proportion of the dataset to include in the test split.
    val_size: Proportion of the training dataset to include in the validation split.
    """
    # Split the dataset into training+validation and test sets
    train_val_input, test_input, train_val_output, test_output = train_test_split(
        input_paths, output_paths, test_size=test_size, random_state=42)

    # Split the training+validation set into training and validation sets
    train_input, val_input, train_output, val_output = train_test_split(
        train_val_input, train_val_output, test_size=val_size, random_state=42)

    return (train_input, train_output), (val_input, val_output), (test_input, test_output)


# Set directory paths and process images
input_directory = 'input'
output_directory = 'output'
processed_input_directory = 'processed_input'
processed_output_directory = 'processed_output'

# Define directory paths for processed images for each dataset
processed_dirs = {
    'train': ('processed_input/train', 'processed_output/train'),
    'val': ('processed_input/val', 'processed_output/val'),
    'test': ('processed_input/test', 'processed_output/test')
}

# Update this with appropriate scale and crop box values
reduce_factor = 0.5  # Adjust this based on requirements
crop_box = (1670, 632, 5400, 3850)  # Adjust this as necessary

# Process images and get batch loaders for each dataset
batch_loaders = check_and_process_images(
    input_directory, output_directory, processed_dirs,
    reduce_factor=reduce_factor, test_size=0.2, val_size=0.25
)


# Model is defined and compiled
EPOCHS = 10  # Adjust based on your requirements
STEPS_PER_EPOCH = 100  # Adjust based on the size of your training dataset and batch size
VALIDATION_STEPS = 20  # Adjust based on the size of your validation dataset and batch size

# Training Loop
for epoch in range(EPOCHS):
    print(f'Epoch {epoch+1}/{EPOCHS}')
    # Train
    for step, (input_batch, output_batch) in enumerate(batch_loaders['train']):
        train_loss, train_accuracy = model.train_on_batch(input_batch, output_batch)
        if step >= STEPS_PER_EPOCH:
            break
    print(f'Training loss: {train_loss}, Training accuracy: {train_accuracy}')

    # Validate
    val_losses, val_accuracies = [], []
    for step, (input_batch, output_batch) in enumerate(batch_loaders['val']):
        val_loss, val_accuracy = model.test_on_batch(input_batch, output_batch)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        if step >= VALIDATION_STEPS:
            break
    avg_val_loss, avg_val_accuracy = np.mean(val_losses), np.mean(val_accuracies)
    print(f'Validation loss: {avg_val_loss}, Validation accuracy: {avg_val_accuracy}')

# Test
test_losses, test_accuracies = [], []
for input_batch, output_batch in batch_loaders['test']:
    test_loss, test_accuracy = model.evaluate(input_batch, output_batch, verbose=0)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
avg_test_loss, avg_test_accuracy = np.mean(test_losses), np.mean(test_accuracies)
print(f'Test loss: {avg_test_loss}, Test accuracy: {avg_test_accuracy}')
