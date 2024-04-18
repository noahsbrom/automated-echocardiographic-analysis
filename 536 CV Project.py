#!/usr/bin/env python
# coding: utf-8

# # Small Dataset CV Project
# 
# 1. Small CNN model trained with available dataset: total trainable parameters = 63 million, optimal val_mape = 8.4 (Epoch 13) or 5.4 within 100 Epochs.
# 2. Feature Extraction with pre-trained Efficient Net B0: total trainable parameters = 6403, optimal val_mape = 8.93 (Epoch 4)
# 3. Fine-tuning a Chest x-ray pre-trained Model: total trainable parameters = 15.2 million, optimal val_mape =  (Epoch )
# 4. Optimal Val_mape is achieved with 20 epochs.
# 5. mean baseline MAPEs:  8.867852813949915 10.000738559848955 8.017046376840236, average: 8.962 (bigger than this number is bad)
# 
# Traing images: 196,
# Validation images: 57,
# Test images: 29.
# 
# Images are resized as 512x512:
# 
# ![image.png](attachment:image.png)
# 

# ## Load and fine-tune the x-ray trained model
# ## Freeze Conv layers, rebuild the output layer and re-train top layers

# In[1]:


# import tensorflow as tf
# base_model= tf.keras.models.load_model('tl_ft_cnn.keras')
# model_config = base_model.get_config()
# model_config["layers"][0]["config"]["batch_input_shape"] = (None, 512, 512, 3)
# x_model = tf.keras.Model.from_config(model_config)

# # Freeze all the layers before the `fine_tune_at` layer and optionally this number should be optimized.
# for layer in x_model.layers[:-10]:  #re-train two conv layers on the top
#     layer.trainable = False
    
# modified_model = tf.keras.models.Sequential()
# for layer in x_model.layers[:-1]: # go through until last layer
#     modified_model.add(layer)

# modified_model.add(tf.keras.layers.Dense(3, name="Output_Layer"))


# In[2]:


import pandas as pd
import os
# file_list = []
# for file in os.listdir('testinput'):
#     if file.endswith('.png'):
#         file_list.append(file)
# df = pd.DataFrame({'Filename': file_list})
# df['Co1']=df.Filename.str[0:3]
# df['Co2']=df.Filename.str[4:7]
# df['Co3']=df.Filename.str[8:11]
# df['ID']=df.Filename.str[0:11]
# df = df.astype({'Co1': 'float', 'Co2': 'float', 'Co3': 'float'})
# df.to_pickle(f"df.pkl")


# In[3]:


df = pd.read_pickle("df.pkl")


# In[4]:


df


# ## CNN Model Training

# In[5]:


from typing import Iterator, List, Union, Tuple, Any
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Model
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import MeanAbsoluteError, MeanAbsolutePercentageError
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import History

# the next 3 lines of code are for my machine and setup due to https://github.com/tensorflow/tensorflow/issues/43174
import tensorflow as tf

#physical_devices = tf.config.list_physical_devices("GPU")
#tf.config.experimental.set_memory_growth(physical_devices[0], True)


def visualize_augmentations(data_generator: ImageDataGenerator, df: pd.DataFrame):
    """Visualizes the keras augmentations with matplotlib in 3x3 grid. This function is part of create_generators() and
    can be accessed from there.

    Parameters
    ----------
    data_generator : Iterator
        The keras data generator of your training data.
    df : pd.DataFrame
        The Pandas DataFrame containing your training data.
    """
    # super hacky way of creating a small dataframe with one image
    series = df.iloc[2]

    df_augmentation_visualization = pd.concat([series, series], axis=1).transpose()

    iterator_visualizations = data_generator.flow_from_dataframe(  # type: ignore
        dataframe=df_augmentation_visualization,
        x_col="image_location",
        y_col=["Co1","Co2","Co3"],
        class_mode="raw",
        target_size=(512, 512),  # size of the image
        batch_size=1,  # use only one image for visualization
    )

    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)  # create a 3x3 grid
        batch = next(iterator_visualizations)  # get the next image of the generator (always the same image)
        img = batch[0]  # type: ignore
        img = img[0, :, :, :]  # remove one dimension for plotting without issues
        plt.imshow(img)
    plt.show()
    plt.close()


def get_mean_baseline(train: pd.DataFrame, val: pd.DataFrame) -> float:
    """Calculates the mean MAE and MAPE baselines by taking the mean values of the training data as prediction for the
    validation target feature.

    Parameters
    ----------
    train : pd.DataFrame
        Pandas DataFrame containing your training data.
    val : pd.DataFrame
        Pandas DataFrame containing your validation data.

    Returns
    -------
    float
        MAPE value.
    """
    y_hat1 = train["Co1"].mean()
    y_hat2 = train["Co2"].mean()
    y_hat3 = train["Co3"].mean()
    val["y_hat1"] = y_hat1
    val["y_hat2"] = y_hat2
    val["y_hat3"] = y_hat3
    mae = MeanAbsoluteError()
    mae1 = mae(val["Co1"], val["y_hat1"]).numpy()
    mae2 = mae(val["Co2"], val["y_hat2"]).numpy()
    mae3 = mae(val["Co3"], val["y_hat3"]).numpy()    # type: ignore
    mape = MeanAbsolutePercentageError()
    mape1 = mape(val["Co1"], val["y_hat1"]).numpy()
    mape2 = mape(val["Co2"], val["y_hat2"]).numpy()
    mape3 = mape(val["Co3"], val["y_hat3"]).numpy()   # type: ignore

    print(mae1, mae2, mae3)
    print("mean baseline MAPEs: ", mape1, mape2, mape3)

    return np.array([mape1,mape2,mape3])


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Accepts a Pandas DataFrame and splits it into training, testing and validation data. Returns DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        Your Pandas DataFrame containing all your data.

    Returns
    -------
    Union[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        [description]
    """
    train, val = train_test_split(df, test_size=0.2, random_state=1)  # split the data with a validation size o 20%
    train, test = train_test_split(
        train, test_size=0.1, random_state=1
    )  # split the data with an overall  test size of 10%

    print("shape train: ", train.shape)  # type: ignore
    print("shape val: ", val.shape)  # type: ignore
    print("shape test: ", test.shape)  # type: ignore

    print("Descriptive statistics of train:")
    print(train.describe())  # type: ignore
    return train, val, test  # type: ignore


def create_generators(
    df: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, plot_augmentations: Any
) -> Tuple[Iterator, Iterator, Iterator]:
    """Accepts four Pandas DataFrames: all your data, the training, validation and test DataFrames. Creates and returns
    keras ImageDataGenerators. Within this function you can also visualize the augmentations of the ImageDataGenerators.

    Parameters
    ----------
    df : pd.DataFrame
        Your Pandas DataFrame containing all your data.
    train : pd.DataFrame
        Your Pandas DataFrame containing your training data.
    val : pd.DataFrame
        Your Pandas DataFrame containing your validation data.
    test : pd.DataFrame
        Your Pandas DataFrame containing your testing data.

    Returns
    -------
    Tuple[Iterator, Iterator, Iterator]
        keras ImageDataGenerators used for training, validating and testing of your models.
    """
    train_generator = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=(0.75, 1),
        shear_range=0.1,
        zoom_range=[0.75, 1],
        horizontal_flip=True,
        validation_split=0.2,
    )  # create an ImageDataGenerator with multiple image augmentations
    validation_generator = ImageDataGenerator(
        rescale=1.0 / 255
    )  # except for rescaling, no augmentations are needed for validation and testing generators
    test_generator = ImageDataGenerator(rescale=1.0 / 255)
    # visualize image augmentations
    if plot_augmentations == True:
        visualize_augmentations(train_generator, df)

    train_generator = train_generator.flow_from_dataframe(
        dataframe=train,
        x_col="image_location",  # this is where your image data is stored
        y_col=["Co1","Co2","Co3"],  # this is your target feature
        class_mode="raw",  # use "raw" for regressions
        target_size=(512, 512),
        batch_size=8,  # increase or decrease to fit your GPU
    )

    validation_generator = validation_generator.flow_from_dataframe(
        dataframe=val, x_col="image_location", y_col=["Co1","Co2","Co3"], class_mode="raw", target_size=(512, 512), batch_size=32,
    )
    test_generator = test_generator.flow_from_dataframe(
        dataframe=test, x_col="image_location", y_col=["Co1","Co2","Co3"], class_mode="raw", target_size=(512, 512), batch_size=32,
    )
    return train_generator, validation_generator, test_generator


def get_callbacks(model_name: str) -> List[Union[TensorBoard, EarlyStopping, ModelCheckpoint]]:
    """Accepts the model name as a string and returns multiple callbacks for training the keras model.

    Parameters
    ----------
    model_name : str
        The name of the model as a string.

    Returns
    -------
    List[Union[TensorBoard, EarlyStopping, ModelCheckpoint]]
        A list of multiple keras callbacks.
    """
    logdir = (
        "logs/scalars/" + model_name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )  # create a folder for each model.
    #    tensorboard_callback = TensorBoard(log_dir=logdir)
    # use tensorboard --logdir logs/scalars in your command line to startup tensorboard with the correct logs

    early_stopping_callback = EarlyStopping(
        monitor="val_mean_absolute_percentage_error",
        min_delta=1,  # model should improve by at least 1%
        patience=10,  # amount of epochs  with improvements worse than 1% until the model stops
        verbose=2,
        mode="min",
        restore_best_weights=True,  # restore the best model with the lowest validation error
    )

    model_checkpoint_callback = ModelCheckpoint(
        "./data/models/" + model_name + ".h5",
        monitor="val_mean_absolute_percentage_error",
        verbose=0,
        save_best_only=True,  # save the best model
        mode="min",
        save_freq="epoch",  # save every epoch
    )  # saving eff_net takes quite a bit of time
    #return [tensorboard_callback, early_stopping_callback, model_checkpoint_callback]
    return [early_stopping_callback, model_checkpoint_callback]

def small_cnn() -> Sequential:
    """A very small custom convolutional neural network with image input dimensions of 224x224x3.

    Returns
    -------
    Sequential
        The keras Sequential model.
    """
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(512, 512, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(3))

    return model


def run_model(
    model_name: str,
    model_function: Model,
    lr: float,
    train_generator: Iterator,
    validation_generator: Iterator,
    test_generator: Iterator,
) -> History:

    callbacks = get_callbacks(model_name)
    model = model_function
    #model.summary()
    plot_model(model, to_file=model_name + ".png", show_shapes=True)

    radam = tfa.optimizers.RectifiedAdam(learning_rate=lr)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    optimizer = ranger

    model.compile(
        optimizer=optimizer, loss="mean_absolute_error", metrics=[MeanAbsoluteError(), MeanAbsolutePercentageError()]
    )
    history = model.fit(
        train_generator,
        epochs=100,
        validation_data=validation_generator,
        callbacks=callbacks,
        workers=1,  # adjust this according to the number of CPU cores of your machine
    )

    model.evaluate(
        test_generator, callbacks=callbacks,
    )
    
    model.save('effnet.keras')
    return history  # type: ignore


def adapt_efficient_net() -> Model:
    """This code uses adapts the most up-to-date version of EfficientNet with NoisyStudent weights to a regression
    problem. Most of this code is adapted from the official keras documentation.

    Returns
    -------
    Model
        The keras model.
    """
    inputs = layers.Input(
        shape=(512, 512, 3)
    )  # input shapes of the images should always be 224x224x3 with EfficientNetB0
    # use the downloaded and converted newest EfficientNet wheights
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="efficientnetb0_notop.h5")
    # Freeze the pretrained weights
    model.trainable = False
    
    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)
    top_dropout_rate = 0.4
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    x = layers.Dense(64, name="last_dense")(x)
    outputs = layers.Dense(3, name="pred")(x)

    # Compile
    model = keras.Model(inputs, outputs, name="EfficientNet")

    return model


def run(small_sample=False):
    """Run all the code of this file.

    Parameters
    ----------
    small_sample : bool, optional
        If you just want to check if the code is working, set small_sample to True, by default False
    """

    df = pd.read_pickle("df.pkl")
    # modify the above file path accordingly.
    
    
    df["image_location"] = (
        "testinput/" + df["ID"] + ".png"
    )  # add the correct path for the image locations.
    
    
    if small_sample == True:
        df = df.iloc[0:1000]  # set small_sampe to True if you want to check if your code works without long waiting
    train, val, test = split_data(df)  # split your data
    mean_baseline = get_mean_baseline(train, val)
    train_generator, validation_generator, test_generator = create_generators(
        df=df, train=train, val=val, test=test, plot_augmentations=True
    )

#     small_cnn_history = run_model(
#         model_name="small_cnn",
#         model_function=small_cnn(),
#         lr=0.001,
#         train_generator=train_generator,
#         validation_generator=validation_generator,
#         test_generator=test_generator,
#     )

    eff_net_history = run_model(
        model_name="eff_net",
        model_function=adapt_efficient_net(),
        lr=0.5,
        train_generator=train_generator,
        validation_generator=validation_generator,
        test_generator=test_generator,
    )
    

    #plot_results(small_cnn_history, eff_net_history, mean_baseline)


# if __name__ == "__main__":
#     run(small_sample=False)


# In[6]:


# new_model = tf.keras.models.load_model("effnet.keras", compile=False)
# new_model.trainable = False

# for layer in new_model.layers[-8:]:
#     layer.trainable = True
#     print(layer)


# In[7]:


df["image_location"] = (
        "testinput/" + df["ID"] + ".png"
    )  # add the correct path for the image locations.


# In[8]:


train, val, test = split_data(df)  # split your data
train_gen, validation_gen, test_gen = create_generators(
    df=df, train=train, val=val, test=test, plot_augmentations=True
)
radam = tfa.optimizers.RectifiedAdam(learning_rate=0.1)
optimizer = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)

# new_model.compile(
#     optimizer=optimizer, loss="mean_absolute_error", metrics=[MeanAbsoluteError(), MeanAbsolutePercentageError()]
# )

# new_model.fit(
#     train_gen,
#     epochs=100,
#     validation_data=validation_gen,
#     workers=1,  # adjust this according to the number of CPU cores of your machine
# )


# In[9]:


# new_model.fit(
#     train_gen,
#     epochs=63,
#     validation_data=validation_gen,
#     workers=1,  # adjust this according to the number of CPU cores of your machine
# )
# new_model.save('effnet_ft.keras')


# In[10]:


# new_model2 = tf.keras.models.load_model("effnet_ft.keras", compile=False)
# new_model2.trainable = False

# for layer in new_model2.layers[-10:]:
#     layer.trainable = True
#     print(layer)


# In[11]:


# new_model2.compile(
#     optimizer=optimizer, loss="mean_absolute_error", metrics=[MeanAbsoluteError(), MeanAbsolutePercentageError()]
# )

# new_model2.fit(
#     train_gen,
#     epochs=100,
#     validation_data=validation_gen,
#     workers=1,  # adjust this according to the number of CPU cores of your machine
# )


# In[12]:


# new_model2.fit(
#     train_gen,
#     epochs=44,
#     validation_data=validation_gen,
#     workers=1,  # adjust this according to the number of CPU cores of your machine
# )
# new_model2.save('effnet_ft2.keras')


# In[13]:


new_model3 = tf.keras.models.load_model("effnet_ft2.keras", compile=False)
new_model3.trainable = False

for layer in new_model3.layers[-18:]:
    layer.trainable = True
    print(layer)
    
new_model3.compile(
    optimizer=optimizer, loss="mean_absolute_error", metrics=[MeanAbsoluteError(), MeanAbsolutePercentageError()]
)

new_model3.fit(
    train_gen,
    epochs=100,
    validation_data=validation_gen,
    workers=1,  # adjust this according to the number of CPU cores of your machine
)


# In[14]:


new_model3.fit(
    train_gen,
    epochs=71,
    validation_data=validation_gen,
    workers=1,  # adjust this according to the number of CPU cores of your machine
)
new_model3.save('effnet_ft3.keras')


# In[15]:


new_model4 = tf.keras.models.load_model("effnet_ft3.keras", compile=False)
new_model4.trainable = False

for layer in new_model4.layers[-21:]:
    layer.trainable = True
    print(layer)
    
new_model4.compile(
    optimizer=optimizer, loss="mean_absolute_error", metrics=[MeanAbsoluteError(), MeanAbsolutePercentageError()]
)

new_model4.fit(
    train_gen,
    epochs=100,
    validation_data=validation_gen,
    workers=1,  # adjust this according to the number of CPU cores of your machine
)


# In[16]:


new_model4.fit(
    train_gen,
    epochs=63,
    validation_data=validation_gen,
    workers=1,  # adjust this according to the number of CPU cores of your machine
)
new_model4.save('effnet_ft4.keras')


# ## Fine-Tuning Complete

# In[29]:


file_list = []
test_dir = "renamedLA/input"
for filename in os.listdir(test_dir):
    if filename.endswith('.png'):
        file_list.append(filename)
df2 = pd.DataFrame({'Filename': file_list})
df2['Co1']=df2.Filename.str[0:3]
df2['Co2']=df2.Filename.str[4:7]
df2['Co3']=df2.Filename.str[8:11]
df2['ID']=df2.Filename.str[0:11]
df2 = df2.astype({'Co1': 'float', 'Co2': 'float', 'Co3': 'float'})
df2["image_location"] = ("renamedLA/input/" + df2["ID"] + ".png")  # add the correct path for the image locations.


# In[ ]:


generator2 = ImageDataGenerator(rescale=1.0 / 255)  # create an ImageDataGenerator with multiple image augmentations

train_generator2 = generator2.flow_from_dataframe(
    dataframe=df2,
    x_col="image_location",  # this is where your image data is stored
    y_col=["Co1","Co2","Co3"],  # this is your target feature
    class_mode="raw",  # use "raw" for regressions
    target_size=(512, 512),
    batch_size=8,  # increase or decrease to fit your GPU
)

new_model = tf.keras.models.load_model("small_cnn.keras", compile=False)
predictions = new_model.predict(train_generator2)
#print(predictions)
# labels = filename.split(",")
# mapes.append(new_model.evaluate(filename, labels))


# mean   384.627551  321.739796  462.693878

# In[ ]:




