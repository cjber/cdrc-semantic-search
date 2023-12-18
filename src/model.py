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

from src.common.utils import Consts, ModelSettings
from src.datastore import CreateDocStore


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
            content = "\n".join([n.get_content() for n in group])
            group[0].node.text = content
            out_nodes.append(group[0])
        return out_nodes


class LlamaIndexModel:
    def __init__(
        self,
        llm: bool,
        top_k: int,
        vector_store_query_mode: str,
        alpha: float,
        prompt=Consts.PROMPT,
    ):
        if llm:
            llm_config = dict(
                model_path="/home/cjber/.cache/huggingface/hub/llama-2-7b-chat.Q4_K_M.gguf",
                temperature=0.1,
                max_new_tokens=256,
                context_window=3900,
                generate_kwargs={},
                model_kwargs={"n_gpu_layers": 50},
                messages_to_prompt=messages_to_prompt,
                completion_to_prompt=completion_to_prompt,
                verbose=True,
            )
            self.llm = LlamaCPP(**llm_config)
        else:
            self.llm = None
        self.prompt = prompt
        self.top_k = top_k
        self.index = self.build_index()

    def run(self, query: str):
        self.response = self.build_response(query)

    def build_index(self):
        service_context = ServiceContext.from_defaults(
            embed_model=HuggingFaceEmbedding(model_name=Consts.HF_EMBED_MODEL),
            llm=self.llm,
        )
        docstore = CreateDocStore()
        docstore.setup_ingestion_pipeline()
        return VectorStoreIndex.from_vector_store(
            docstore.vector_store,
            service_context=service_context,
            show_progress=True,
        )

    def build_response(self, query):
        text_qa_template = PromptTemplate(self.prompt)

        if self.llm:
            retriever = self.index.as_query_engine(
                text_qa_template=text_qa_template,
                response_mode="accumulate",
                vector_store_query_mode="hybrid",
                alpha=0.5,
                similarity_top_k=self.top_k,
                node_postprocessors=[DocumentGroupingPostprocessor()],
            )
            self.response = retriever.query(query)
        else:
            retriever = self.index.as_retriever(
                vector_store_query_mode=self.vector_store_query_mode,
                alpha=self.alpha,
                similarity_top_k=self.top_k,
            )
            self.response = retriever.retrieve(query)
        return self.process_response(self.response)

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
    model_settings = ModelSettings.parse_file("./config/model.json")
    model = LlamaIndexModel(**model_settings.model_dump())
    model.run("inequality")
    print(model.response[0]["response"])
