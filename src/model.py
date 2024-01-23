import logging
import os
import sys
from statistics import mean
from typing import Optional

from llama_index import ServiceContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingMode
from llama_index.indices.query.schema import QueryBundle
from llama_index.llms import LlamaCPP, OpenAI
from llama_index.llms.llama_utils import (
    completion_to_prompt,
    messages_to_prompt,
)
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.prompts import PromptTemplate
from llama_index.response import Response
from llama_index.schema import NodeWithScore

from src.common.utils import Settings
from src.datastore import CreateDataStore

logging.basicConfig(level=logging.ERROR, filename="logs/model.log", filemode="w")
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


class DocumentGroupingPostprocessor(BaseNodePostprocessor):
    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle],
    ) -> list[NodeWithScore]:
        nodes_by_document = {}

        for node in nodes:
            document_id = node.metadata["id"]
            if document_id not in nodes_by_document:
                nodes_by_document[document_id] = []
            nodes_by_document[document_id].append(node)

        out_nodes = []
        for group in nodes_by_document.values():
            content = "\n--------------------\n".join([n.get_content() for n in group])
            # score = mean([n.score for n in group])
            score = max(n.score for n in group)
            group[0].node.text = content
            group[0].score = score
            out_nodes.append(group[0])
        return out_nodes


class LlamaIndexModel:
    def __init__(
        self,
        top_k: int,
        vector_store_query_mode: str,
        alpha: float,
        prompt: str,
        response_mode: str,
        load_model: bool = True,
    ):
        self.model = OpenAI() if load_model else None
        self.top_k = top_k
        self.vector_store_query_mode = vector_store_query_mode
        self.alpha = alpha
        self.prompt = prompt
        self.response_mode = response_mode

        self.index = self.build_index()

    def run(self, query: str):
        self.query = query

        self.response = self.build_response()
        self.processed_response = self.process_response(self.response)

    def build_index(self):
        self.service_context = ServiceContext.from_defaults(
            embed_model=OpenAIEmbedding(
                mode=OpenAIEmbeddingMode.SIMILARITY_MODE,
                api_key=os.environ["OPENAI_API_KEY"],
            ),
            llm=self.model,
        )
        docstore = CreateDataStore(**Settings().datastore.model_dump())
        docstore.setup_ingestion_pipeline()
        return VectorStoreIndex.from_vector_store(
            docstore.vector_store,
            service_context=self.service_context,
            show_progress=True,
            use_async=True,
        )

    def build_response(self):
        retriever = self.index.as_retriever(
            vector_store_query_mode=self.vector_store_query_mode,
            alpha=self.alpha,
            similarity_top_k=self.top_k,
        )
        response = retriever.retrieve(self.query)
        postprocessor = DocumentGroupingPostprocessor()
        response = postprocessor.postprocess_nodes(response)
        return response

    @staticmethod
    def process_response(response):
        if isinstance(response, list):
            scores = [r.score for r in response]
            out = [r.node.metadata for r in response]
            for item in out:
                item["score"] = scores.pop(0)

        elif isinstance(response, Response):
            out = {"response": response.response}
            out = [out, response.metadata]
        return out

    def explain_dataset(self, response_num: int):
        if not self.response:
            raise ValueError("No response to explain")
        # text_qa_template = PromptTemplate(self.prompt)

        response = self.response[response_num]
        query_engine = self.index.as_query_engine(nodes=response)
        response = query_engine.query(self.query)
        self.explained_response = response.response


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR, filename="logs/model.log", filemode="w")

    model = LlamaIndexModel(**Settings().model.model_dump())
    model.run("diabetes")
    model.explain_dataset(1)
