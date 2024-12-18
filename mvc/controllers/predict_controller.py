from fastapi import APIRouter, File, UploadFile
from models.prediction import predict_image

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Handles the POST request to predict the plant disease."""
    image_data = await file.read()
    prediction = predict_image(image_data)
    return prediction
