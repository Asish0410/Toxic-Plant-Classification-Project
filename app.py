from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from sklearn.utils import class_weight  # Import class_weight
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}  # Added gif
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define image size
img_height = 224
img_width = 224

# Load the trained model
#  Important:  Load the model *after* defining img_height and img_width
model = load_model('plant_classification_model.h5')  # Changed model name to the one you provided

# Class names - replace with your actual class names.  Important:  Order must match your training data.
class_names = ['Non-Poisonous', 'Poisonous']  #<--- IMPORTANT ORDER



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



def preprocess_image(img_path, target_size): #made a function
    """Loads and preprocesses an image for the model."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale
    return img_array



def predict_plant(img_path, model, class_names):
    """
    Predicts whether a plant is poisonous or non-poisonous based on an image.
    Args:
        img_path: Path to the image file.
        model: Trained tensorflow model.
        class_names: List of class names.
    Returns:
        A tuple containing the predicted class and the model's confidence.
    """
    img_array = preprocess_image(img_path, (img_height, img_width)) #use function
    prediction = model.predict(img_array)
    confidence = prediction[0][0]
    predicted_class = class_names[int(np.round(prediction)[0][0])]
    return predicted_class, np.float32(confidence)



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction="No file uploaded")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction="No file selected")

        if file and allowed_file(file.filename):
            # Secure the filename and save the file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # Make prediction
                predicted_class, confidence = predict_plant(filepath, model, class_names)

                return render_template(
                    'index.html',
                    prediction=predicted_class,
                    confidence=confidence,
                    image_path=os.path.join('uploads', filename)  # Relative path for HTML
                )
            except Exception as e:
                return render_template('index.html', prediction=f"Error: {str(e)}")
        else:
            return render_template('index.html', prediction="Invalid file type. Please upload a valid image.")

    return render_template('index.html')  # Render the form for GET requests



if __name__ == '__main__':
    app.run(debug=True)
