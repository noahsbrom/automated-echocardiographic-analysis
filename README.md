# automated-echocardiographic-analysis

For the model to work you need the original input (/input) and output (/output) folders with all the 283 tiff files (LA measurements).

To start:

- For preprocessing the images and extracting the coordinates, run in cmd: python preprocess.py

- After that you can train (cmd python model.py), but the model is already pretrained in the directory: test.h5

- Then you can test the line drawing based on the model, by running: python draw.py

