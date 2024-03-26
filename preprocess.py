import os
import cv2
import numpy as np
from PIL import Image
import csv

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█', printEnd="\r"):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    if iteration == total:
        print()

def crop_and_save_as_png(image_path, output_directory):
    print(f"Processing image: {image_path}")
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = image[y:y+h, x:x+w]
        resized_image = cv2.resize(cropped_image, (512, 512), interpolation=cv2.INTER_AREA)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        new_filename = base_name + ".png"
        new_image_path = os.path.join(output_directory, new_filename)
        cv2.imwrite(new_image_path, resized_image)
    else:
        print(f"No contours found in image {image_path}")

    return new_image_path

def find_endpoints(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image {img_path}")
        return None

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([0,50,50])
    upper_blue = np.array([130,255,255])
    mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        annotation_contour = contours[0]
        x, y, w, h = cv2.boundingRect(annotation_contour)
        return [str(x + w // 2), str(y), str(y + h)]
    else:
        return None

def extract_coordinates_and_save(image_path, output_dir):
    print(f"Extracting endpoints from image: {image_path}")
    coordinates = find_endpoints(image_path)
    if coordinates:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        txt_filename = base_name + "_coordinates.txt"
        txt_path = os.path.join(output_dir, txt_filename)
        with open(txt_path, 'w') as file:
            file.write(','.join(coordinates))
    else:
        print(f"No valid endpoints found in image {image_path}")


def process_input_directory(input_dir, testoutput_dir):
    ensure_directory_exists(input_dir)
    ensure_directory_exists(testoutput_dir)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.tiff') or f.lower().endswith('.tif')]
    total_files = len(files)
    print(f"Processing {total_files} TIFF files from input directory.")
    for i, filename in enumerate(files, start=1):
        f = os.path.join(input_dir, filename)
        if os.path.isfile(f):
            print_progress_bar(i, total_files, prefix='Processing Input:', suffix='Complete', length=50)
            crop_and_save_as_png(f, testoutput_dir)

def process_output_directory_and_save_coordinates(output_dir, testoutput_dir, temp_dir):
    ensure_directory_exists(output_dir)
    ensure_directory_exists(testoutput_dir)  # Make sure the directory for saving coordinates exists.
    files = [f for f in os.listdir(output_dir) if f.lower().endswith('.tiff') or f.lower().endswith('.tif')]
    total_files = len(files)
    print(f"Processing {total_files} TIFF files from output directory for coordinates extraction.")

    for i, filename in enumerate(files, start=1):
        original_file_path = os.path.join(output_dir, filename)
        if os.path.isfile(original_file_path):
            print_progress_bar(i, total_files, prefix='Extracting Coordinates:', suffix='Complete', length=50)

            # Crop, resize, and save the new PNG image; also capture the new image path
            # crop_and_save_as_png should be modified to return the path of the new image
            new_image_path = crop_and_save_as_png(original_file_path, temp_dir)

            # Proceed to extract coordinates only if the new image was successfully created
            if new_image_path:
                # Extract and save endpoints from the newly created PNG image
                # Adjusted to use the resized image for endpoint extraction
                extract_coordinates_and_save(new_image_path, testoutput_dir)



def main():
    input_dir = 'input'
    testoutput_dir = 'testoutput'
    output_dir = 'output_original'
    temp_dir = 'temp'

    print("Processing input directory...")
    process_input_directory(input_dir, testoutput_dir)
    print("Processing output directory and extracting coordinates...")
    process_output_directory_and_save_coordinates(output_dir, testoutput_dir, temp_dir)  # Fixed this line
    print("Processing complete.")


if __name__ == "__main__":
    main()
