from io import BytesIO
from typing import List
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from model import load_model, predict, prepare_cli
#from PIL import Image
from pydantic import BaseModel
import json
import pandas as pd
from zipfile import ZipFile

app = FastAPI()
model = load_model()
# Define the response JSON

class Prediction(BaseModel):
    #content_type: str
    Pourcentage_de_non_solvabilité: int
    #Pourcentage_de_solvabilité: int
@app.post("/predict/", response_model=Prediction)
def prediction(id):
    # Ensure that the file is an image
    #if not file.content_type.startswith("image/"):
    #    raise HTTPException(status_code=400, detail="File provided is not an image.")
    #content = await file.read()
    z = ZipFile("X_sample.zip")
    data = pd.read_csv(z.open('X_sample.csv'), encoding ='utf-8')
    #data = pd.read_csv('X_sample.csv')
    chk_id = id
    data2 = data[data['SK_ID_CURR'] == int(chk_id)]
    #js = data2.to_json()
    #content = js
    
    #cli = json.loads(js)
        
    #df = pd.read_json(cli, orient ='columns')
    # preprocess the image and prepare it for classification
    #cli = prepare_cli(content)
    
    sample = data2.drop('SK_ID_CURR', axis=1)
    response = predict(sample, chk_id, model)
    #response2 = predict2(sample, chk_id, model)
    # return the response as a JSON
    return { 
    #"content_type": str(data2), }
    "Pourcentage_de_non_solvabilité" : response, }
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000)
