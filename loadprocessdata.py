# Loading data for machine learning model
# Created for LA images
# Martinus Kleiweg

import numpy as np
import os
from PIL import Image
from tqdm import tqdm

def load_image_paths(directory):
    file_paths = []
    filenames = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            file_path = os.path.join(directory, filename)
            file_paths.append(file_path)
            filenames.append(filename)
    return file_paths, filenames

def save_processed_images(file_paths, save_directory, reduce_factor=None):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    for file_path in tqdm(file_paths, desc="Processing and saving images"):
        with Image.open(file_path) as img:
            img = img.convert('L')
            if reduce_factor:
                original_size = img.size
                new_size = (int(original_size[0] * reduce_factor), int(original_size[1] * reduce_factor))
                img = img.resize(new_size, Image.ANTIALIAS)
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

def check_and_process_images(input_directory, output_directory, processed_input_directory, processed_output_directory, reduce_factor=None):
    if os.path.exists(processed_input_directory) and os.listdir(processed_input_directory) and os.path.exists(processed_output_directory) and os.listdir(processed_output_directory):
        print("Processed directories exist and contain images. Proceed to batch loading...")
        # Note: Returning generator functions instead of actual data
        return (
            batch_load_saved_images(processed_input_directory),
            batch_load_saved_images(processed_output_directory)
        )
    else:
        print("Processed directories do not exist or are empty. Processing and saving images...")
        input_paths, input_filenames = load_image_paths(input_directory)
        output_paths, output_filenames = load_image_paths(output_directory)
        assert input_filenames == output_filenames, "Mismatch between input and output filenames"
        save_processed_images(input_paths, processed_input_directory, reduce_factor=reduce_factor)
        save_processed_images(output_paths, processed_output_directory, reduce_factor=reduce_factor)
        return (
            batch_load_saved_images(processed_input_directory),
            batch_load_saved_images(processed_output_directory)
        )

# Set directory paths and process images
input_directory = 'input'
output_directory = 'output'
processed_input_directory = 'processed_input'
processed_output_directory = 'processed_output'
reduce_factor = 0.5  # Image resize factor, adjust this based on requirements

input_batches, output_batches = check_and_process_images(
    input_directory, output_directory, processed_input_directory, processed_output_directory,
    reduce_factor=reduce_factor
)

# Example of how we might use the batches
for input_batch, output_batch in zip(input_batches, output_batches):
    # Here we would feed each batch into your model for training, prediction, etc.
    pass
