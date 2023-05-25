import pandas as pd
from fastapi import FastAPI, File, HTTPException
from models import ChurnLinearModel, ChurnDecisionTree
from io import BytesIO


app = FastAPI()
models_path = {
    'linear': (ChurnLinearModel, 'linear_model.pickle'),
    'decision_tree': (ChurnDecisionTree, 'decision_tree.pickle')
}
columns = [
    'accountlength', 'areacode', 'customerservicecalls',
    'numbervmailmessages', 'totaldaycalls', 'totaldayminutes',
    'totalevecalls', 'totaleveminutes', 'totalintlcalls',
    'totalintlminutes', 'totalnightcalls', 'totalnightminutes'
]
models = {}


@app.on_event("startup")
def load_models():
    for model_name, (model_type, model_checkpoint) in models_path.items():
        new_model = model_type()
        new_model.load(model_checkpoint)
        models[model_name] = new_model


@app.get("/")
def root():
    return {"message": "Hello! Application is running"}


@app.post("/predict/")
async def predict(model_name: str, file: bytes = File()):
    if model_name not in models:
        raise HTTPException(status_code=404, detail='Unknown model')

    model = models[model_name]

    try:
        data = pd.read_csv(BytesIO(file))
        data = data.reindex(sorted(data.columns), axis=1)
        assert list(data.columns) == columns
    except:
        raise HTTPException(status_code=404, detail='Incorrect file format')

    try:
        predictions = model.predict(data).tolist()
        predictions = [str(prediction) for prediction in predictions]
        probabilities = model.predict_proba(data)[:, 1].tolist()
    except:
        raise HTTPException(status_code=500, detail='Internal server error')

    return {'predictions': predictions, 'probabilities': probabilities}
