from fastapi import FastAPI

from src.common.utils import Settings
from src.model import LlamaIndexModel


def create_app():
    app = FastAPI()
    model = LlamaIndexModel(**Settings().model.model_dump())
    return app, model


app, model = create_app()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/query")
async def query(q: str):
    model.run(q)
    return model.processed_response
