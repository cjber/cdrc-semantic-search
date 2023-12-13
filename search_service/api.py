from fastapi import FastAPI

from src.datastore import pipeline

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/query")
async def query(query):
    return pipeline(query=query)
