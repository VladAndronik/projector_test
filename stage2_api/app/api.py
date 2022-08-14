import json

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder

from model.predict import make_prediction

api_router = APIRouter()


@api_router.post("/predict", status_code=200)
async def predict(input_data):
    results = make_prediction(input_data)
    return results
