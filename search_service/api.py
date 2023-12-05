import os

from fastapi import FastAPI
from haystack.document_stores import PineconeDocumentStore
from haystack.nodes import DensePassageRetriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline

document_store = PineconeDocumentStore(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment="gcp-starter",
    similarity="dot_product",
    embedding_dim=768,
    index="cdrc",
    recreate_index=False,
)
retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    use_gpu=True,
    embed_title=True,
)
reader = FARMReader(model_name_or_path="deepset/bert-base-cased-squad2", use_gpu=True)

pipeline = ExtractiveQAPipeline(reader, retriever)

app = FastAPI()


@app.get("/query")
async def query(q):
    return pipeline.run(query=q)
