# Automated Echocardiographic Analysis Tool
Some information about the app

## Starting the app
1. Ensure that python is installed
2. Ensure that the Tesseract-OCR Engine is installed. See for instructions below.
3. Run `./deploy.sh` from the app directory. 
4. The app should now be running on localhost:8080. 

## Understanding the app
* The deploy script configures a virtual environment in which
the user can run the app without installing various dependecies on their
machine.
* Image uploads are stored in static/image-uploads, a directory that is created 
on deployment and deleted on termination. By default, Flask serves static files
(like CSS stylesheets and image uploads) from a static view that takes a path relative
to the app/static directory.

## Installing the Tesseract-OCR Engine
For Windows:
1. Download the installer from the official Tesseract GitHub page.
2. Run the installer and follow the instructions. It’s important to note the installation path.
3. Add Tesseract’s installation path to your system’s PATH environment variable. This is typically C:\Program Files\Tesseract-OCR if you used the default install location.

For macOS:
1. You can install Tesseract using Homebrew: brew install tesseract

For Linux (Ubuntu/Debian):
You can install Tesseract using apt-get:
1. sudo apt-get update
2. sudo apt-get install tesseract-ocr
3. sudo apt-get install libtesseract-dev
