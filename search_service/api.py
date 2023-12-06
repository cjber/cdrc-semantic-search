import os

from fastapi import FastAPI
from haystack.document_stores import PineconeDocumentStore
from haystack.nodes import EmbeddingRetriever, FARMReader, TableTextRetriever
from haystack.pipelines import ExtractiveQAPipeline

app = FastAPI()


def haystack_pipeline():
    document_store = PineconeDocumentStore(
        api_key=os.environ.get("PINECONE_API_KEY"),
        environment="gcp-starter",
        similarity="dot_product",
        embedding_dim=768,
        index="cdrc",
        recreate_index=False,
    )
    retriever = TableTextRetriever(document_store=document_store, use_gpu=True)
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        use_gpu=True,
    )
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

    pipeline = ExtractiveQAPipeline(reader, retriever)
    return pipeline


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/query")
async def query(q):
    return haystack_pipeline().run(query=q)
