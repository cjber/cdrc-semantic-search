from fastapi import FastAPI

from src.common.utils import Settings
from pydantic import BaseModel
from src.model import LlamaIndexModel
from uuid import UUID, uuid4


class ResponsePost(BaseModel):
    results_id: UUID = uuid4()


def create_app():
    app = FastAPI()
    model = LlamaIndexModel(**Settings().model.model_dump())
    return app, model


app, model = create_app()


@app.get("/")
def index():
    return {"message": "Make a post request to /query."}


@app.post("/query")
async def query(q: str) -> list[dict]:
    model.run(q)
    return [{"results_id": uuid4()}] + model.processed_response


@app.get("/explain/{results_id}")
async def explain(response_num: int, results_id: UUID) -> dict:
    model.explain_dataset(response_num)
    return {
        "results_id": results_id,
        "explained_response": model.explained_response,
        "related_dataset": model.processed_response[response_num],
    }


@app.get("/results/{results_id}")
async def results(results_id: UUID) -> list[dict]:
    return [{"results_id": results_id}] + model.processed_response


@app.post("/query_str/{results_id}")
async def query_str(results_id: UUID) -> dict:
    return {"results_id": results_id, "query": model.query}
