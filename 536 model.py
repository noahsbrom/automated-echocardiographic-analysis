#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Define paths to dataset folders
input_images_path = 'path_to_input_images'
output_images_path = 'path_to_output_images'

# Function to load images and convert them to arrays
def load_images(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".tif"): # or .jpg, or whatever file type you're using
            img_path = os.path.join(folder, filename)
            img = load_img(img_path, color_mode='rgb', target_size=(256, 256)) # change target_size to match your image dimensions
            img = img_to_array(img)
            images.append(img)
    return np.array(images)

# Load the input and output images
input_images = load_images(input_images_path)
output_images = load_images(output_images_path)

# Normalize the pixel values to [0, 1]
input_images = input_images.astype('float32') / 255.
output_images = output_images.astype('float32') / 255.

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(input_images, output_images, test_size=0.1, random_state=42)

# Build the CNN model
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)))
model.add(MaxPooling2D((2, 2), padding='same'))
# Add more convolutional layers, pooling layers, and eventually upsampling layers
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same')) # Change the number of filters to match the number of channels in the output image

# Compile the model
model.compile(optimizer=Adam(), loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, shuffle=True, validation_data=(X_val, y_val))

# Save the model
model.save('line_annotation_model.h5')


# Some points to consider:
# 
# Model Architecture: The given architecture is very basic. Depending on your specific task, you might need a more complex architecture, like a U-Net, which is designed for tasks like segmentation.
# 
# Normalization: It’s important to normalize the input and output images in the same way.
# 
# Loss Function: 'mean_squared_error' is used for simplicity, but you may need a more sophisticated loss function that better captures the accuracy of the line placement.
# 
# Epochs and Batch Size: These parameters should be tuned. You might need more epochs or a different batch size depending on the complexity of the task and the size of your dataset.
# 
# Validation Split: We are holding out 10% of the data for validation; you may adjust this percentage based on your dataset size.
# 
# This is a very high-level and simplified version of what you would need to do. The actual implementation would likely require additional steps, including but not limited to, more advanced data augmentation, hyperparameter tuning, implementing callbacks for monitoring the training process, and potentially using transfer learning with pre-trained models for better feature extraction.

# In[ ]:


# U-Net model
def unet(input_size=(256,256,3)):
    inputs = Input(input_size)
    
    # Downsampling
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # ... (additional downsampling blocks)
    
    # Bottleneck
    # ...

    # Upsampling
    # ...
    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv7))
    merge8 = concatenate([conv1,up8], axis=3)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

model = unet()

# Print model summary to verify architecture
print(model.summary())

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=2, verbose=1, validation_data=(X_val, y_val))

# Save the model
model.save('unet_annotation_model.h5')


# Please note the following:
# 
# Architecture: The actual U-Net architecture includes symmetric downsampling and upsampling paths with skip connections. You'll need to complete the downsampling (encoding) path, the bottleneck, and the upsampling (decoding) path.
# 
# Output Layer: The output layer uses a sigmoid activation function, which is common for binary segmentation tasks. If your annotations can be represented as binary masks (where the line is one class, and everything else is another class), this is appropriate.
# 
# Loss Function: This example uses binary_crossentropy as the loss function since we're treating the problem as a binary segmentation task. Depending on your specific requirements, you may need to choose a different loss function.
# 
# Learning Rate and Optimizer: The learning rate and optimizer are critical hyperparameters that you might need to adjust.
# 
# Batch Size and Epochs: Depending on the size of your images and the memory capacity of your machine, you may need to adjust the batch size. Similarly, the number of epochs may need to be increased or decreased based on the performance of the model during training.
# 
# Model Saving: The model is saved after training so it can be used later for inference without needing to be retrained.
# 
# Remember to complete the missing parts of the U-Net architecture. You might want to refer to the original U-Net paper or other resources for the specifics of the architecture. Also, this is a simplified representation, and for practical purposes, you would add more robust training practices like callbacks for early stopping, model checkpointing, and potentially more sophisticated data augmentation techniques.

# In[ ]:


def unet(input_size=(256,256,3)):
    inputs = Input(input_size)
    
    # Downsampling
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # ... (additional downsampling blocks)
    
    # Bottleneck
    # ...

    # Upsampling
    # ...
    up8 = UpSampling2D(size=(2,2))(conv7)
    merge8 = concatenate([conv1,up8], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    
    conv9 = Conv2D(2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

model = unet(input_size=(256,256,3))

# Callbacks
earlystopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
modelcheckpoint = ModelCheckpoint('unet_best_model.h5', monitor='val_loss', save_best_only=True, verbose=1, mode='min')
reducelronplateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='min', min_lr=1e-6)

# Train the model
history = model.fit(
    X_train, y_train, 
    validation_data=(X_val, y_val), 
    epochs=100,  # Increase if necessary
    batch_size=2,  # Adjust based on the GPU memory
    verbose=1,
    callbacks=[earlystopping, modelcheckpoint, reducelronplateau]
)

# Load the best model
model.load_model('unet_best_model.h5')

# Save the final model
model.save('unet_final_model.h5')


# A few important notes:
# 
# Callbacks:
# 
# EarlyStopping stops training when a monitored metric has stopped improving.
# ModelCheckpoint saves the model after every epoch where the validation loss has improved.
# ReduceLROnPlateau reduces the learning rate when a metric has stopped improving.
# Model Loading: Before making predictions on new images, load the best model saved by the ModelCheckpoint.
# 
# Missing Parts: The above code includes placeholders where you need to add additional downsampling and upsampling blocks. The U-Net architecture typically includes several blocks of convolutional layers followed by a max-pooling layer for downsampling, and then corresponding upsampling layers followed by convolutional layers.
# 
# Adjustments: Depending on the actual size of your images, you might need to adjust the size of the model

# In[ ]:


def conv_block(input_tensor, num_filters, kernel_size, dropout_rate, padding='same', activation=tf.nn.relu):
    x = tf.keras.layers.Conv2D(num_filters, kernel_size, padding=padding, activation=activation)(input_tensor)
    x = tf.keras.layers.Conv2D(num_filters, kernel_size, padding=padding, activation=activation)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x

def upconv_block(input_tensor, concat_tensor, num_filters, kernel_size, dropout_rate, padding='same', activation=tf.nn.relu):
    x = tf.keras.layers.Conv2DTranspose(num_filters, kernel_size, strides=2, padding=padding)(input_tensor)
    x = tf.keras.layers.Concatenate()([x, concat_tensor])
    x = conv_block(x, num_filters, kernel_size, dropout_rate, activation=activation)
    return x

def build_unet(input_shape, num_filters_list):
    inputs = tf.keras.layers.Input(input_shape)

    # Encoder
    encoder_block1 = conv_block(inputs, num_filters_list[0], 3, 0.3)
    encoder_block2 = conv_block(encoder_block1, num_filters_list[1], 3, 0.3)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(encoder_block2)

    encoder_block3 = conv_block(pool1, num_filters_list[2], 3, 0.3)
    encoder_block4 = conv_block(encoder_block3, num_filters_list[3], 3, 0.3)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(encoder_block4)

    encoder_block5 = conv_block(pool2, num_filters_list[4], 3, 0.3)
    encoder_block6 = conv_block(encoder_block5, num_filters_list[5], 3, 0.3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(encoder_block6)

    # Bottleneck
    bottleneck = conv_block(pool3, num_filters_list[6], 3, 0.3)

    # Decoder
    upconv1 = upconv_block(bottleneck, encoder_block6, num_filters_list[5], 2, 0.3)
    upconv2 = upconv_block(upconv1, encoder_block4, num_filters_list[4], 2, 0.3)
    upconv3 = upconv_block(upconv2, encoder_block3, num_filters_list[3], 2, 0.3)
    upconv4 = upconv_block(upconv3, encoder_block2, num_filters_list[2], 2, 0.3)
    upconv5 = upconv_block(upconv4, encoder_block1, num_filters_list[1], 2, 0.3)

    # Output layer
    output_layer = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(upconv5)

    model = tf.keras.Model(inputs=inputs)

