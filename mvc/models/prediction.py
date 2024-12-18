import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image

# Load the model
MODEL = tf.keras.models.load_model('model.keras', compile=False)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def read_file_as_image(data) -> np.ndarray:
    """Converts the uploaded file into an image array."""
    image = np.array(Image.open(BytesIO(data)))
    return image

def predict_image(image_data) -> dict:
    """Predicts the disease based on the uploaded image."""
    image = read_file_as_image(image_data)
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }
