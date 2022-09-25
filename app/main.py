from io import BytesIO
from typing import List
from fastapi import FastAPI, File, HTTPException, UploadFile
import uvicorn
from model import _load_model, predict,prepare_data
from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()
model = _load_model()
# Define the response JSON
class Prediction(BaseModel):
    filename: str
    content_type: str
    predictions: float
@app.post("/predict", response_model=Prediction)
async def prediction(file: UploadFile = File(...)):
    # Ensure that the file is an image
    data = await file.read()
    print(data)
    res = prepare_data(data)
    
    response = predict(res, model)
    # return the response as a JSON
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "predictions": response,
    }
@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"message": str(exc)},
    )

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000)