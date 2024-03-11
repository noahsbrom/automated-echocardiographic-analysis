from PIL import Image
import numpy as np
from tqdm import tqdm
import os

# Set directories here
input_directory = 'input'  # Path to the directory containing original echocardiogram images.
output_directory = 'output'  # Path to the directory where binary masks will be saved.
annotation_directory = 'output_original'  # Path to the directory containing annotated images.

# Set other parameters
final_size = (512, 512)  # The size to which images and masks should be resized
target_color = (4, 253, 255)  # The RGB value of the annotation color
tolerance = 30  # Color tolerance

def create_color_based_binary_masks(input_directory, output_directory, annotation_directory, final_size, target_color, tolerance):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    filenames = sorted(os.listdir(annotation_directory))
    for filename in tqdm(filenames, desc="Creating binary masks"):
        if filename.endswith('.tif') or filename.endswith('.tiff'):
            # Load the annotated image
            annotation_path = os.path.join(annotation_directory, filename)
            with Image.open(annotation_path) as annotated_img:
                # Convert image to RGBA
                annotated_img = annotated_img.convert('RGBA')
                # Prepare binary mask
                binary_mask = Image.new('L', annotated_img.size, 0)  # Create a new black image
                pixels = np.array(annotated_img)
                # Exclude the alpha channel
                pixels = pixels[..., :3]  # Keep only R, G, and B channels
                # Identify pixels within the color range
                mask = np.all(np.abs(pixels - target_color) <= tolerance, axis=-1)
                # Update binary mask
                binary_mask.putdata((255 * mask).astype(np.uint8).flatten())
                # Resize image and mask to match model input
                binary_mask = binary_mask.resize(final_size, Image.Resampling.LANCZOS)
                # Save the binary mask
                save_path = os.path.join(output_directory, os.path.splitext(filename)[0] + '.png')
                binary_mask.save(save_path, 'PNG')

create_color_based_binary_masks(input_directory, output_directory, annotation_directory, final_size, target_color, tolerance)
