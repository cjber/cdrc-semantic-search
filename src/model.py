from statistics import mean
from typing import Optional

from llama_index import ServiceContext, VectorStoreIndex
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.indices.query.schema import QueryBundle
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import (
    completion_to_prompt,
    messages_to_prompt
)
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.prompts import PromptTemplate
from llama_index.response import Response
from llama_index.schema import NodeWithScore
from pydantic import Field
from pydantic_settings import BaseSettings

from src.common.utils import Config, Settings
from src.datastore import CreateDataStore


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
            content = "\n\n".join([n.get_content() for n in group])
            score = mean([n.score for n in group])
            group[0].node.text = content
            group[0].score = score
            out_nodes.append(group[0])
        return out_nodes


class LlamaIndexModel:
    def __init__(
        self,
        top_k: int,
        hf_embed_model: str,
        vector_store_query_mode: str,
        alpha: float,
        prompt: str,
    ):
        llm_config = dict(
            n_ctx=3900,
            n_threads=8,
            n_gpu_layers=35,
        )
        self.model = LlamaCPP(
            model_path="/home/cjber/.cache/huggingface/hub/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            temperature=0.1,
            max_new_tokens=256,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            model_kwargs=llm_config,
            verbose=True,
        )
        self.top_k = top_k
        self.hf_embed_model = hf_embed_model
        self.vector_store_query_mode = vector_store_query_mode
        self.alpha = alpha
        self.prompt = prompt

        self.index = self.build_index()

    def run(self, query: str, use_llm=True):
        self.response = self.build_response(query, use_llm)

    def build_index(self):
        service_context = ServiceContext.from_defaults(
            embed_model=HuggingFaceEmbedding(model_name=self.hf_embed_model),
            llm=self.model,
        )
        docstore = CreateDataStore(
            **Settings().datastore.model_dump(),
            **Settings().shared.model_dump(),
        )
        docstore.setup_ingestion_pipeline()
        return VectorStoreIndex.from_vector_store(
            docstore.vector_store,
            service_context=service_context,
            show_progress=True,
        )

    def build_response(self, query, use_llm=True):
        text_qa_template = PromptTemplate(self.prompt)

        if use_llm:
            retriever = self.index.as_query_engine(
                text_qa_template=text_qa_template,
                response_mode="accumulate",
                vector_store_query_mode=self.vector_store_query_mode,
                alpha=self.alpha,
                similarity_top_k=self.top_k,
                node_postprocessors=[DocumentGroupingPostprocessor()],
            )
            response = retriever.query(query)
        else:
            retriever = self.index.as_retriever(
                vector_store_query_mode=self.vector_store_query_mode,
                alpha=self.alpha,
                similarity_top_k=self.top_k,
            )
            response = retriever.retrieve(query)
            postprocessor = DocumentGroupingPostprocessor()
            response = postprocessor.postprocess_nodes(response)
        return self.process_response(response)

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


if __name__ == "__main__":
    model = LlamaIndexModel(
        **Settings().model.model_dump(),
        **Settings().shared.model_dump(),
    )
    model.run("diabetes", use_llm=True)
