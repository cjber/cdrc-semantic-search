import json
import dateparser
import logging
import os
import shutil
from pathlib import Path
from pinecone import Pinecone, PodSpec

from llama_hub.file.unstructured import UnstructuredReader
from llama_index import SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingMode
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
        if main_id == catalogue_meta["id"]:
            return {
                "title": catalogue_meta["title"],
                "id": catalogue_meta["id"],
                "url": catalogue_meta["url"],
                "date_created": dateparser.parse(
                    catalogue_metadata[0]["metadata_created"]
                ).isoformat(),
            }


class CreateDataStore:
    def __init__(
        self,
        index_name: str,
        chunk_size: int,
        chunk_overlap: int,
        overwrite: bool,
        embed_dim: int,
        profiles_dir: str = Paths.PROFILES_DIR,
        data_dir: str = Paths.DATA_DIR,
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
        reader = UnstructuredReader(
            api=True, api_key=os.environ["UNSTRUCTURED_API_KEY"]
        )
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
        self.vector_store = PineconeVectorStore(
            self.pc.Index(
                self.index_name,
                host="https://cdrc-index-afz2q2b.svc.gcp-starter.pinecone.io",
            )
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
    logging.basicConfig(
        level=logging.ERROR, filename="logs/datastore.log", filemode="w"
    )

    datastore = CreateDataStore(**Settings().datastore.model_dump())
    datastore.run()
