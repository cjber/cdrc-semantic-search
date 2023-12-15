from llama_index import (
    DocumentSummaryIndex,
    ServiceContext,
    StorageContext,
    VectorStoreIndex
)
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import (
    completion_to_prompt,
    messages_to_prompt
)
from llama_index.prompts import PromptTemplate
from llama_index.response import Response

from src.common.utils import Consts
from src.datastore import setup_ingestion_pipeline


def build_index(llm: bool):
    # model_url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
    if llm:
        llm = LlamaCPP(
            # model_url=model_url,
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
    else:
        llm = None

    pipeline = setup_ingestion_pipeline()
    service_context = ServiceContext.from_defaults(
        embed_model=HuggingFaceEmbedding(model_name=Consts.HF_EMBED_MODEL),
        llm=llm,
    )
    return VectorStoreIndex.from_vector_store(
        pipeline.vector_store,
        service_context=service_context,
        show_progress=True,
    )


def build_response(index, query, llm: bool = True):
    text_qa_template_str = (
        "Summarise the following information in under 50 words. Following this summary suggest a link with the users 'query', Using your own knowledge or the documents.\n"
        "Query: {query_str}\n"
        "Documents:\n"
        "\n---------------------\n{context_str}\n---------------------\n"
        "For each unique dataset, structure your answer as follows:\n\n"
        "Dataset: <dataset name>\n"
        "Summary: <dataset summary>\n"
        "Link: <dataset link with query>\n\n"
    )
    text_qa_template = PromptTemplate(text_qa_template_str)

    if llm:
        query_engine = index.as_query_engine(
            text_qa_template=text_qa_template,
            vector_store_query_mode="hybrid",
            alpha=0.5,
            similarity_top_k=6,
        )
        res = query_engine.query(query)
    else:
        query_engine = index.as_retriever(
            vector_store_query_mode="hybrid",
            alpha=0.5,
            similarity_top_k=10,
        )
        res = query_engine.retrieve(query)
    return res


def process_response(res):
    if isinstance(res, list):
        scores = [r.score for r in res]
        out = [r.node.metadata for r in res]
        for item in out:
            item["score"] = scores.pop(0)

    elif isinstance(res, Response):
        response = {"response": res.response}
        out = [response, res.metadata]
    return out


def main():
    llm = False
    index = build_index(llm=llm)
    res = build_response(index, "diabetes", llm=llm)
    res = process_response(res)
    return res


if __name__ == "__main__":
    res = main()
