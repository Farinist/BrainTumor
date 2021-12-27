from flask import Flask, render_template, request
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model

app = Flask(__name__)
dict = {0 : 'Tumor Otak Glioma', 1 : 'No Tumor', 2 : 'Tumor Otak Meningioma', 3 : 'Tumor Otak Pituitari'}
model = tf.keras.models.load_model('braintumor.h5')
model.make_predict_function()

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    return render_template('index.html')

@app.route('/submit', methods=['GET','POST'])
def predict():
    imagefile=request.files['imagefile']
    image_path = "static/" + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(150,150))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    classes = model.predict(image)

    if np.argmax(classes[0])==0:
     p='Tumor Otak Glioma'
    elif np.argmax(classes[0])==1:
     p='Normal'
    elif np.argmax(classes[0])==2:
     p='Tumor Otak Meningioma'
    else:
     p='Tumor Otak Pituitari'
    
    classification = '%s' % (p)

    return render_template('index.html', prediction=classification, image=image_path)

if __name__ == '__main__':
    app.run(port=3000, debug=True)