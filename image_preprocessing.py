#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load the image
import cv2
import numpy as np

def crop_black_edges(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to get regions that are not completely black
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours from the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour based on area
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle for the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the image using the bounding rectangle of the largest contour
    cropped_image = image[y:y+h, x:x+w]

    # Save the cropped image
    # Change to your preferred output file path
    cv2.imwrite(image_path, cropped_image)

def find_endpoints(img_path):
    img = cv2.imread(img_path)

    # Convert the image to HSV (Hue, Saturation, Value) color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the range of blue color in HSV
    # These values can be adjusted to get the specific shade of blue in the annotations
    lower_blue = np.array([0,50,50])
    upper_blue = np.array([130,255,255])

    # Create a mask for blue color
    mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the annotation line will be one of the longest detected contours
    # Sort the contours by length and get the longest one
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    annotation_contour = contours[0]

    # Get bounding box coordinates of the contour
    x, y, w, h = cv2.boundingRect(annotation_contour)

    # Vertical line
    x1 = str(x + w // 2)
    y1 = str(y)
    y2 = str(y + h)
    
    return [x1,y1,y2]


# In[2]:


# directory = 'input'

# for filename in os.listdir(directory):
#     f = os.path.join(directory, filename)
#     # checking if it is a file
#     if os.path.isfile(f):
#         crop_black_edges(f)
    
# directory2 = 'output'

# for filename in os.listdir(directory2):
#     f = os.path.join(directory2, filename)
#     # checking if it is a file
#     if os.path.isfile(f):
#         crop_black_edges(f)


# In[3]:


# final steps to polishing up directory
import os
from PIL import Image

input_directory = 'testinput'
output_directory = 'testoutput'

#filenames = []
for filename in os.listdir(output_directory):
    f1 = os.path.join(output_directory, filename)
    f2 = os.path.join(input_directory, filename)
    if os.path.isfile(f1):
        # find the endpoints
        endpoints = find_endpoints(f1)
#         # resize the images
#         img1 = Image.open(f1)
#         new_img1 = img1.resize((512,512))
#         img2 = Image.open(f2)
#         new_img2 = img2.resize((512,512))
        # create new filename
        new_filename = ",".join(endpoints)
        new_filename = new_filename + ".png"
        
        #filenames.append(new_filename)
        f3 = os.path.join(output_directory, new_filename)
        f4 = os.path.join(input_directory, new_filename)
        os.rename(f1, f3)
        os.rename(f2, f4)


# In[ ]:




