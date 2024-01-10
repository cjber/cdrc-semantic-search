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
async def query(q: str, use_llm: bool):
    model.run(q, use_llm)
    return model.processed_response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
