import os

import matplotlib.pyplot as plt
from fastapi import FastAPI
from haystack.document_stores import PineconeDocumentStore
from haystack.nodes import (
    EmbeddingRetriever,
    FARMReader,
    PromptNode,
    PromptTemplate
)
from haystack.pipelines import ExtractiveQAPipeline, Pipeline

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
    # retriever = TableTextRetriever(document_store=document_store, use_gpu=True)
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        use_gpu=True,
    )
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

    pipeline = ExtractiveQAPipeline(reader, retriever)
    return pipeline


def gpt_pipeline():
    prompt = PromptTemplate(
        prompt="""
The following is a query to a search system and a retrived dataset. Explain in detail how the dataset is relevant to the query.

Query: "{query}"

Dataset: \n\n"{to_strings(documents, pattern='Title: $title' + new_line + new_line + '$content', str_replace={new_line: ' ', '[': '(', ']': ')'})}"

Explanation:
""",
    )

    document_store = PineconeDocumentStore(
        api_key=os.environ.get("PINECONE_API_KEY"),
        environment="gcp-starter",
        similarity="dot_product",
        embedding_dim=768,
        index="cdrc",
        recreate_index=False,
    )
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        use_gpu=True,
    )
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
    prompt_node = PromptNode(
        model_name_or_path="google/flan-t5-large",
        output_variable="explanation",
        max_length=512,
        model_kwargs={
            "stream": False,
            "model_max_length": 1024,
            "model_min_length": 512,
            "length_penalty": 2,
            "num_beams": 16,
            "no_repeat_ngram_size": 2,
            "early_stopping": True,
        },
        default_prompt_template=prompt,
        use_gpu=False,
    )

    pipeline = Pipeline()
    pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
    pipeline.add_node(component=reader, name="Reader", inputs=["Retriever"])
    pipeline.add_node(component=prompt_node, name="Prompt", inputs=["Reader"])
    return pipeline


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/query")
async def query(q):
    return gpt_pipeline().run(query=q)
