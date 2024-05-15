from uuid import UUID, uuid4

from fastapi import Depends, FastAPI

from src.common.utils import Settings
from src.model import LlamaIndexModel

app = FastAPI()
model_instances = {}
query_mapping = {}


def get_model(results_id: UUID) -> LlamaIndexModel:
    if results_id not in model_instances:
        model_instances[results_id] = LlamaIndexModel(**Settings().model.model_dump())
    return model_instances[results_id]


@app.get("/")
def index():
    return {"message": "Make a post request to /query."}


@app.post("/query")
async def query(q: str) -> dict:
    results_id = uuid4()
    query_mapping[results_id] = q
    return {"results_id": results_id, "query": q}


@app.get("/results/{results_id}")
async def results(
    results_id: UUID, model: LlamaIndexModel = Depends(get_model)
) -> dict:
    if results_id not in query_mapping:
        return {"error": "No query found for the provided results_id"}

    q = query_mapping[results_id]
    model.run(q)
    return {
        "results_content": model.processed_response,
        "metadata": {
            "results_id": results_id,
            "query": q,
        },
    }


@app.get("/explain/{results_id}")
async def explain(
    response_num: int, results_id: UUID, model: LlamaIndexModel = Depends(get_model)
) -> dict:
    if results_id not in query_mapping:
        return {"error": "No query found for the provided results_id"}

    model.explain_dataset(response_num)
    return {
        "explained_response": model.explained_response,
        "metadata": {
            "results_id": results_id,
            "query": query_mapping[results_id],
            "related_dataset": model.processed_response[response_num],
        },
    }
