from fastapi import FastAPI

from src.common.utils import ModelSettings
from src.model import LlamaIndexModel


def create_app():
    model_settings = ModelSettings.parse_file("./config/model.json")
    model = LlamaIndexModel(**model_settings.model_dump())
    app = FastAPI()
    return app, model


app, model = create_app()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/query")
async def query(q):
    model.run(q)
    return model.response
