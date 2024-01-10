import json
import logging
import os
import shutil
from pathlib import Path

import pinecone
from llama_hub.file.unstructured import UnstructuredReader
from llama_index import SimpleDirectoryReader
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.ingestion import IngestionPipeline
from llama_index.text_splitter import SentenceSplitter
from llama_index.vector_stores import PineconeVectorStore

from src.common.utils import Paths, Settings


def _add_metadata_to_document(doc_id: str) -> dict:
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

    for catalogue_meta in catalogue_metadata:
        # NOTE: Add date information to metadata
        if main_id == catalogue_meta["id"]:
            return {
                "title": catalogue_meta["title"],
                "id": catalogue_meta["id"],
                "url": catalogue_meta["url"],
            }


class CreateDataStore:
    def __init__(
        self,
        index_name: str,
        hf_embed_model: str,
        hf_embed_dim: int,
        chunk_size: int,
        chunk_overlap: int,
        overwrite: bool,
        profiles_dir: str = Paths.PROFILES_DIR,
        data_dir: str = Paths.DATA_DIR,
        pipeline_storage: Path = Paths.PIPELINE_STORAGE,
    ):
        self.index_name = index_name
        self.overwrite = overwrite
        self.hf_embed_dim = hf_embed_dim
        self.hf_embed_model = hf_embed_model
        self.profiles_dir = profiles_dir
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.pipeline_storage = pipeline_storage

    def run(self):
        self.initialise_pinecone_index()
        self.setup_directory_reader()
        self.setup_ingestion_pipeline()
        self.load_and_preprocess_documents()

        shutil.rmtree(self.profiles_dir)

    def initialise_pinecone_index(self):
        pinecone.init(
            api_key=os.environ["PINECONE_API_KEY"],
            environment=os.environ["PINECONE_ENVIRONMENT"],
        )
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                self.index_name,
                dimension=self.hf_embed_dim,
                metric="cosine",
            )
        elif self.overwrite:
            pinecone.delete_index(self.index_name)
            pinecone.create_index(
                self.index_name,
                dimension=self.hf_embed_dim,
                metric="cosine",
            )

    def setup_directory_reader(self):
        reader = UnstructuredReader(api_key=os.environ["UNSTRUCTURED_API_KEY"])
        self.dir_reader = SimpleDirectoryReader(
            self.profiles_dir,
            recursive=True,
            file_extractor={
                ".pdf": reader,
                ".docx": reader,
                ".txt": reader,
            },
            file_metadata=lambda name: _add_metadata_to_document(Path(name).stem),
        )

    def setup_ingestion_pipeline(self):
        self.pinecone_index = pinecone.Index(self.index_name)
        self.vector_store = PineconeVectorStore(self.pinecone_index)
        self.pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                ),
                HuggingFaceEmbedding(model_name=self.hf_embed_model),
            ],
            vector_store=self.vector_store,
        )

    def load_and_preprocess_documents(self):
        self.docs = self.dir_reader.load_data(show_progress=True)
        for doc in self.docs:
            doc.excluded_embed_metadata_keys.extend(["id", "url", "filename"])
            doc.excluded_llm_metadata_keys.extend(["id", "url", "filename"])

        self.pipeline.run(documents=self.docs)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.ERROR, filename="logs/datastore.log", filemode="w"
    )

    datastore = CreateDataStore(**Settings().datastore.model_dump())
    datastore.run()
