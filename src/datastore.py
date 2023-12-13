import os
from pathlib import Path

import pinecone
from llama_index import (
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    download_loader
)
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.extractors import (
    BaseExtractor,
    EntityExtractor,
    KeywordExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    TitleExtractor
)
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import (
    completion_to_prompt,
    messages_to_prompt
)
from llama_index.prompts import PromptTemplate
from llama_index.text_splitter import TokenTextSplitter
from llama_index.vector_stores import PineconeVectorStore

from src.common.utils import _add_metadata_to_document


def build_service_context():
    model_url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
    llm = LlamaCPP(
        model_url=model_url,
        temperature=0.1,
        max_new_tokens=256,
        context_window=3900,
        generate_kwargs={},
        model_kwargs={"n_gpu_layers": 50},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )

    embed_model = HuggingFaceEmbedding()
    text_splitter = TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=128)
    service_context = ServiceContext.from_defaults(
        embed_model=embed_model,
        llm=llm,
        transformations=[text_splitter],
    )
    return service_context


def create_vector_store(profiles_dir: Path, index_name: str = "cdrc-index"):
    if index_name not in pinecone.list_indexes():
        pinecone.init(
            api_key=os.environ["PINECONE_API_KEY"],
            environment=os.environ["PINECONE_ENVIRONMENT"],
        )
        pinecone.create_index(index_name, dimension=384, metric="cosine")

        UnstructuredReader = download_loader("UnstructuredReader")
        dir_reader = SimpleDirectoryReader(
            profiles_dir,
            recursive=True,
            file_extractor={
                ".pdf": UnstructuredReader(),
                ".docx": UnstructuredReader(),
                ".txt": UnstructuredReader(),
            },
            file_metadata=lambda name: _add_metadata_to_document(Path(name).stem),
        )
        docs = dir_reader.load_data()
        docs[0]
        service_context = build_service_context()
        VectorStoreIndex.from_documents(docs, service_context=service_context)

    pinecone_index = pinecone.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index)
    return vector_store


def init_model(vector_store):
    service_context = build_service_context()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        service_context=service_context,
        show_progress=True,
    )
    return index


def build_response(index, query, llm=True):
    template = (
        "We have provided the following descriptions of datasets. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Using this context, please explain why each are relevant to the following query. Use one sentence. Do not repond as if I have asked you a question, just describe the dataset. For each dataset structure the reply using 'Dataset:' <dataset name>\n\n 'Response:\n\n<why the dataset is relevent>'\n\n Query: {query_str}\n"
    )

    prompt = PromptTemplate(template)
    if llm:
        query_engine = index.as_query_engine(text_qa_template=prompt)
        res = query_engine.query(query)
    elif llm is None:
        query_engine = index.as_retriever(text_qa_template=prompt)
        res = query_engine.retrieve(query)

    return res


def pipeline(query):
    profiles_dir = Path("./data/profiles")
    vector_store = create_vector_store(profiles_dir, index_name="cdrc-index")
    index = init_model(vector_store=vector_store)
    res = build_response(index, query=query)
    return res


if __name__ == "__main__":
    res = pipeline(query="test")
