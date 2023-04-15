import pandas as pd 
import numpy as np
import uvicorn
#http://127.0.0.1:8000/docs.
from fastapi import FastAPI, File, UploadFile, Form
from io import StringIO
from src.PredictService import PredictService
import io
from fastapi.responses import StreamingResponse
app = FastAPI()
import json


@app.post('/Batch Method/')
async def create_data_file(
        data_file: UploadFile = File(...),
        ):
    csv=pd.read_csv("../data/waste.csv",sep=';')
    #csv=pd.read_csv(StringIO(str(data_file.file.read(), 'utf-8')), encoding='utf-8')
    prediction= PredictService(pathToModelHard='C:/Users/juano/Documents/HACKATONS/DatafestIKEA2023/DataFestImportBilly/API/models/modelo_el_00075.pkl',pathToModelSoft='C:/Users/juano/Documents/HACKATONS/DatafestIKEA2023/DataFestImportBilly/API/models/modelo_el_005.pkl')
    solution=prediction.predict(csv)
    stream = io.StringIO()
    solution.to_csv(stream, index = False)
    response = StreamingResponse(iter([stream.getvalue()]),
                                 media_type="text/csv"
                                )
    response.headers["Content-Disposition"] = "attachment; filename=export.csv"
    return response
@app.post("/Real Time/")
def create_upload_files(upload_file: UploadFile = File(...)):
    json_data = json.load(upload_file.file)
    return {"data_in_file": json_data}
@app.post('/uploadfile/')
async def create_data_file(
        experiment: str = Form(...),
        file_type: str = Form(...),
        file_id: str = Form(...),
        data_file: UploadFile = File(...),
        ):
    
    #decoded = base64.b64decode(data_file.file)
    #decoded = io.StringIO(decoded.decode('utf-8'))
    
    print(pd.read_csv(data_file.file, sep='\t'))

    return {'filename': data_file.filename, 
            'experiment':experiment, 
            'file_type': file_type, 
            'file_id': file_id}