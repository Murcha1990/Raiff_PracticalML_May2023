from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/text/")
async def process_text(text: str):
    return {"processed_text": text.upper()}


@app.get("/replace/")
async def replace_chars(text: str = 'abracadabra', src: str = 'a', dst: str = 'o'):
    return {"processed_text": text.replace(src, dst)}
