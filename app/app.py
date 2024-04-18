from flask import Flask, render_template, request, redirect, session
import os
import secrets

app = Flask(__name__)

# Generate a random string of 32 bytes
app.secret_key = secrets.token_hex(32)

@app.route('/') 
def home():
    return render_template('index.html')


"""
@app.route('/uploaded', methods=['POST'])
def upload_image():
    image = request.files['image']
    if image:
        user_image_path = f'./static/image-uploads/{image.filename}'
        session['image_to_process'] = user_image_path
        image.save(user_image_path)
        return render_template('uploaded.html', image_path = user_image_path)
    return redirect('/')
"""

@app.route('/uploaded', methods=['POST'])
def upload_image():
    images = images = request.files.getlist('image')
    image_paths = []
    if images:
        for image in images:
            user_image_path = f'./static/image-uploads/{image.filename}'
            image_paths.append(user_image_path)
            image.save(user_image_path)
        session['images_to_process'] = image_paths
        return render_template('uploaded.html', image_paths=image_paths)
    return redirect('/')


@app.route('/processed')
def process_image():
    image_paths = session.get('images_to_process')
    ## TODO: update with backend image processing
    return f'path to local images: {image_paths}'
    


