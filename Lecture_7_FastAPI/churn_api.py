import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query
from models import ChurnLinearModel
from io import BytesIO


app = FastAPI()
linear_model_path = 'linear_model.pickle'
columns = [
    'accountlength', 'areacode', 'customerservicecalls',
    'numbervmailmessages', 'totaldaycalls', 'totaldayminutes',
    'totalevecalls', 'totaleveminutes', 'totalintlcalls',
    'totalintlminutes', 'totalnightcalls', 'totalnightminutes'
]
linear_model: ChurnLinearModel | None = None


@app.on_event("startup")
def load_model():
    global linear_model
    linear_model = ChurnLinearModel()
    linear_model.load(linear_model_path)


@app.get("/")
def root():
    return {"message": "Hello! Application is running"}


@app.get("/str/")
async def linear_model_str(query: str):
    query = query.split(',')
    if not len(query) == len(columns):
        raise HTTPException(status_code=400, detail='Incorrect number of features')

    try:
        data = {column: [float(value)] for column, value in zip(sorted(columns), query)}
        data = pd.DataFrame(data)
    except:
        raise HTTPException(status_code=400, detail='Data non-numerical')

    try:
        prediction = linear_model.predict(data)[0]
        probability = linear_model.predict_proba(data)[0, 1]
    except:
        raise HTTPException(status_code=500, detail='Internal server error')

    return {"prediction": str(prediction), "probability": str(probability)}


@app.get("/list/")
async def linear_model_list(query: list[float] = Query(default=None)):
    if len(query) != len(columns):
        raise HTTPException(
            status_code=400,
            detail=f'Wrong number of features is selected! Required {len(columns)}, got {len(query)}'
        )

    data = {name: [value] for name, value in zip(sorted(columns), query)}
    data = pd.DataFrame(data)
    try:
        prediction = linear_model.predict(data)[0]
        probability = linear_model.predict_proba(data)[0, 1]
    except:
        raise HTTPException(status_code=500, detail='Internal server error')

    return {'prediction': str(prediction), 'probability': probability}


@app.post("/file/")
async def linear_model_file(file: bytes = File()):
    try:
        data = pd.read_csv(BytesIO(file))
        data = data.reindex(sorted(data.columns), axis=1)
        assert list(data.columns) == columns
    except:
        raise HTTPException(status_code=400, detail='Incorrect file format')

    try:
        predictions = linear_model.predict(data).tolist()
        predictions = [str(prediction) for prediction in predictions]
        probabilities = linear_model.predict_proba(data)[:, 1].tolist()
    except:
        raise HTTPException(status_code=500, detail='Internal server error')

    return {'predictions': predictions, 'probabilities': probabilities}
