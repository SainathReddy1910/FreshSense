#Imports required for the model to predict and display results app 
#import pickle
import os

import numpy as np
from PIL import Image
# Imports required for the flask app
from flask import Flask, flash, request, redirect, url_for, render_template
from keras.models import load_model
from keras.utils import img_to_array, load_img
from werkzeug.utils import secure_filename



app = Flask(__name__, template_folder='templates', static_folder='static')
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#Resizing the image given by the user
def resize_ima(im,new_width):
    height,width = im.size
    ratio = height/width
    new_height = int(ratio*new_width)
    new_in = im.resize((new_width,new_height))
    return new_in

#Home route
@app.route('/')
def home():
    return render_template('predict.html')

@app.route('/homepage')
def homepage():
    return render_template('homef.html')

@app.route('/facts')
def facts():
    return render_template('fact.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file=Image.open(file)
        file=resize_ima(file,500)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)
        prediction = predict_one(image_path)
        #accu = show_acc()
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('predict.html', filename=filename,a=prediction)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

def predict_one(image_path):
    mod = load_model('fresh1(1).h5')
    filename=image_path
    classes = ['Fresh','Fresh','Fresh','Rotten','Rotten','Rotten']
    test_image = load_img(filename,target_size=(64,64))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = mod.predict(test_image)
    result1 = result[0]
    for i in range(6):
            if result1[i] == 1.:
                break;
    prediction = classes[i]
    return prediction

if __name__== '__main__':
    app.run(debug=True)


