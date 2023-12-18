from fastapi import FastAPI

from src.model import LlamaIndexModel


def create_app():
    model = LlamaIndexModel()
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
