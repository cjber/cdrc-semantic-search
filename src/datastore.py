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


class CreateDocStore:
    def __init__(
        self,
        index_name: str = Consts.INDEX_NAME,
        overwrite: bool = True,
        embed_dim: int = Consts.EMBED_DIM,
        embed_model: str = Consts.HF_EMBED_MODEL,
        profiles_dir: str = Paths.PROFILES_DIR,
        data_dir: str = Paths.DATA_DIR,
        chunk_size: int = 256,
        chunk_overlap: int = 0,
        pipeline_storage: Path = Paths.PIPELINE_STORAGE,
    ):
        self.index_name = index_name
        self.overwrite = overwrite
        self.embed_dim = embed_dim
        self.embed_model = embed_model
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

    def initialise_pinecone_index(self):
        pinecone.init(
            api_key=os.environ["PINECONE_API_KEY"],
            environment=os.environ["PINECONE_ENVIRONMENT"],
        )
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                self.index_name,
                dimension=self.embed_dim,
                metric="cosine",
            )
        elif self.overwrite:
            pinecone.delete_index(self.index_name)
            pinecone.create_index(
                self.index_name,
                dimension=self.embed_dim,
                metric="cosine",
            )

    def setup_directory_reader(self):
        self.dir_reader = SimpleDirectoryReader(
            self.profiles_dir,
            recursive=True,
            file_extractor={
                ".pdf": UnstructuredReader(),
                ".docx": UnstructuredReader(),
                ".txt": UnstructuredReader(),
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
                HuggingFaceEmbedding(model_name=self.embed_model),
            ],
            vector_store=self.vector_store,
            docstore=SimpleDocumentStore(),
        )

    def load_and_preprocess_documents(self):
        self.docs = self.dir_reader.load_data(show_progress=True)
        for doc in self.docs:
            doc.excluded_embed_metadata_keys.extend(["id", "url", "filename"])
            doc.excluded_llm_metadata_keys.extend(["id", "url", "filename"])

        if self.pipeline_storage.exists() and not self.overwrite:
            self.pipeline.load(self.pipeline_storage)
        self.pipeline.run(documents=self.docs)
        self.pipeline.persist(self.pipeline_storage)


if __name__ == "__main__":
    docstore = CreateDocStore(overwrite=True)
    docstore.run()
