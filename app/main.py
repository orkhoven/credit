from io import BytesIO
from typing import List
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from model import load_model, predict, prepare_cli
#from PIL import Image
from pydantic import BaseModel
import json
import pandas as pd


app = FastAPI()
model = load_model()
# Define the response JSON

class Prediction(BaseModel):
    Pourcentage_de_non_solvabilité: int
    #Pourcentage_de_solvabilité: int
@app.post("/predict", response_model=Prediction)
async def prediction(file: UploadFile = File(...)):
    # Ensure that the file is an image
    #if not file.content_type.startswith("image/"):
    #    raise HTTPException(status_code=400, detail="File provided is not an image.")
    content = await file.read()
    cli = json.loads(content)
        
    df = pd.read_json(cli)
    # preprocess the image and prepare it for classification
    #cli = prepare_cli(content)
    #chk_id = df['SK_ID_CURR']
    sample = df 
    response = predict(sample, model)
    #response2 = predict2(sample, chk_id, model)
    # return the response as a JSON
    return { "Pourcentage_de_non_solvabilité": response }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000)
