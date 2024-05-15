import json
import logging
import os
import shutil
from pathlib import Path

import dateparser
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingMode
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_parse import LlamaParse
from pinecone import Pinecone, PodSpec

from src.common.utils import Paths, Settings


def _add_metadata_to_document(doc_id: str) -> dict[str, str]:
    with open(Paths.DATA_DIR / "catalogue-metadata.json") as f:
        catalogue_metadata = json.load(f)
    with open(Paths.DATA_DIR / "files-metadata.json") as f:
        files_metadata = json.load(f)

    format, main_id = doc_id.split("-", maxsplit=1)

    if format != "notes":
        for file_meta in files_metadata:
            if main_id == file_meta["id"]:
                main_id = file_meta["parent_id"]
                break

    iso_date = dateparser.parse(files_metadata[0]["created"]).isoformat()  # type: ignore
    for cm in catalogue_metadata:
        if main_id == cm["id"]:
            return {
                "title": cm["title"],
                "id": cm["id"],
                "url": cm["url"],
                "date_created": iso_date,
            }
    raise ValueError(f"Metadata not found for document {doc_id}")


class CreateDataStore:
    def __init__(
        self,
        index_name: str,
        chunk_size: int,
        chunk_overlap: int,
        overwrite: bool,
        embed_dim: int,
        profiles_dir: Path = Paths.PROFILES_DIR,
        data_dir: Path = Paths.DATA_DIR,
        pipeline_storage: Path = Paths.PIPELINE_STORAGE,
    ):
        self.index_name = index_name
        self.overwrite = overwrite
        self.profiles_dir = profiles_dir
        self.data_dir = data_dir
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.pipeline_storage = pipeline_storage

        self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    def run(self):
        if not self.profiles_dir.exists():
            return logging.error(
                f"Profiles directory {self.profiles_dir} does not exist."
            )
        self.initialise_pinecone_index()
        self.setup_directory_reader()
        self.setup_ingestion_pipeline()
        self.load_and_preprocess_documents()

        shutil.rmtree(self.profiles_dir)

    def initialise_pinecone_index(self):
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embed_dim,
                metric="cosine",
                spec=PodSpec(environment=os.environ["PINECONE_ENVIRONMENT"]),
            )
        elif self.overwrite:
            self.pc.delete_index(self.index_name)
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embed_dim,
                metric="cosine",
                spec=PodSpec(environment=os.environ["PINECONE_ENVIRONMENT"]),
            )

    def setup_directory_reader(self):
        pdf_reader = LlamaParse()
        self.dir_reader = SimpleDirectoryReader(
            str(self.profiles_dir),
            recursive=True,
            file_extractor={".pdf": pdf_reader},
            file_metadata=lambda name: _add_metadata_to_document(Path(name).stem),
        )

    def setup_ingestion_pipeline(self):
        self.vector_store = PineconeVectorStore(
            pinecone_index=self.pc.Index(self.index_name)
        )
        self.pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                ),
                OpenAIEmbedding(
                    mode=OpenAIEmbeddingMode.TEXT_SEARCH_MODE,
                    model="text-embedding-3-large",
                    api_key=os.environ["OPENAI_API_KEY"],
                ),
            ],
            vector_store=self.vector_store,
        )

    def load_and_preprocess_documents(self):
        self.docs = self.dir_reader.load_data(show_progress=True)
        for doc in self.docs:
            doc.excluded_embed_metadata_keys.extend(
                ["id", "url", "filename", "date_created"]
            )
            doc.excluded_llm_metadata_keys.extend(
                ["id", "url", "filename", "date_created"]
            )

        self.pipeline.run(documents=self.docs)


if __name__ == "__main__":
    datastore = CreateDataStore(**Settings().datastore.model_dump())
    datastore.run()
