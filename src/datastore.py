import os
from pathlib import Path

import pinecone
from llama_index import SimpleDirectoryReader, download_loader
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.ingestion import IngestionPipeline
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.text_splitter import SentenceSplitter
from llama_index.vector_stores import PineconeVectorStore

from src.common.utils import Consts, Paths, _add_metadata_to_document

UnstructuredReader = download_loader("UnstructuredReader")


def initialise_pinecone_index():
    if Consts.INDEX_NAME not in pinecone.list_indexes():
        pinecone.init(
            api_key=os.environ["PINECONE_API_KEY"],
            environment=os.environ["PINECONE_ENVIRONMENT"],
        )
        pinecone.create_index(
            Consts.INDEX_NAME,
            dimension=Consts.EMBED_DIM,
            metric="cosine",
        )


def setup_directory_reader():
    return SimpleDirectoryReader(
        Paths.PROFILES_DIR,
        recursive=True,
        file_extractor={
            ".pdf": UnstructuredReader(),
            ".docx": UnstructuredReader(),
            ".txt": UnstructuredReader(),
        },
        file_metadata=lambda name: _add_metadata_to_document(Path(name).stem),
    )


def setup_ingestion_pipeline():
    pinecone_index = pinecone.Index(Consts.INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index)

    return IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=256),
            HuggingFaceEmbedding(model_name=Consts.HF_EMBED_MODEL),
        ],
        vector_store=vector_store,
        docstore=SimpleDocumentStore(),
    )


def load_and_preprocess_documents(dir_reader, pipeline):
    docs = dir_reader.load_data(show_progress=True)
    for doc in docs:
        doc.excluded_embed_metadata_keys.extend(["id", "url", "filename"])
        doc.excluded_llm_metadata_keys.extend(["id", "url", "filename"])

    if Paths.PIPELINE_STORAGE.exists():
        pipeline.load(Paths.PIPELINE_STORAGE)
    pipeline.run(documents=docs)
    return pipeline


def main():
    initialise_pinecone_index()
    dir_reader = setup_directory_reader()
    pipeline = setup_ingestion_pipeline()
    pipeline = load_and_preprocess_documents(dir_reader, pipeline)
    pipeline.persist(Paths.PIPELINE_STORAGE)


if __name__ == "__main__":
    main()
