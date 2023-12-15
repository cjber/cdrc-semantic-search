from fastapi import FastAPI

from src.common.utils import Paths
from src.model import build_index, build_response, process_response


def create_app():
    index = build_index(llm=True)
    app = FastAPI()
    return app, index


app, index = create_app()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/query")
async def query(q):
    res = build_response(index, q, llm=True)
    return process_response(res)
