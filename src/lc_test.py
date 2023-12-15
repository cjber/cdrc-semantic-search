import os
from pathlib import Path

from langchain.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

loader = DirectoryLoader(
    Path("./data/profiles"),
    recursive=True,
    use_multithreading=True,
    show_progress=True,
    loader_cls=UnstructuredFileLoader,
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

docsearch = Pinecone.from_documents(docs, HuggingFaceEmbeddings(), index_name="cdrc-index")
