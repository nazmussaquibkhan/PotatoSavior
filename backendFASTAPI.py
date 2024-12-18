from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()

# Add CORS middleware to the app.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your model (make sure the path is correct)
MODEL = tf.keras.models.load_model('model.keras', compile=False)

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/")
async def main():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image and prepare it for the model
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    # Debugging: print input image shape and raw prediction
    print(f"Input image shape: {img_batch.shape}")
    predictions = MODEL.predict(img_batch)

    # Debugging: print prediction values
    print(f"Predicted probabilities: {predictions[0]}")

    # Get the predicted class and confidence
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    # Return class and confidence
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='localhost', port=8000)
