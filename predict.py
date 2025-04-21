import tensorflow as tf
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your trained deepfake detection model
model = tf.keras.models.load_model('model/model.h5')

# Class labels
class_names = ['Real', 'Fake']

def predict_image(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction
    predictions = model.predict(img_array)
    
    # If the model outputs a single sigmoid value
    if predictions.shape[-1] == 1:
        predicted_class = int(predictions[0][0] > 0.5)
        confidence = predictions[0][0] if predicted_class == 1 else 1 - predictions[0][0]
    else:
        # If the model uses softmax with 2 output classes
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]

    result = class_names[predicted_class]
    return f"{result} ({confidence * 100:.2f}%)"
