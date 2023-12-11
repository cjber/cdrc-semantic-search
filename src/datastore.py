import json
import os
from argparse import ArgumentParser

from elasticsearch_haystack.document_store import ElasticsearchDocumentStore
from elasticsearch_haystack.embedding_retriever import (
    ElasticsearchEmbeddingRetriever
)
from haystack import Pipeline
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder
)
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores import DuplicatePolicy, InMemoryDocumentStore
from unstructured_fileconverter_haystack import UnstructuredFileConverter

from src.common.utils import Paths, _add_metadata_to_document, validate_files

parser = ArgumentParser()
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite the existing index",
)

args = parser.parse_args()


def _remove_old_docs(document_store):
    docs_remove, notes_remove, docs_missing, notes_missing = validate_files()

    if docs_remove:
        document_store.delete_documents(filters={"id": {"$in": docs_remove}})
    if notes_remove:
        document_store.delete_documents(filters={"id": {"$in": notes_remove}})


def _missing_docs(document_store):
    with open(Paths.DATA_DIR / "files-metadata.json", "r") as f:
        file_data = json.load(f)
    with open(Paths.DATA_DIR / "catalogue-metadata.json", "r") as f:
        catalogue_data = json.load(f)

    file_ids = {file["id"] for file in file_data}
    catalogue_ids = {catalogue["id"] for catalogue in catalogue_data}
    docs = {doc.to_dict()["meta"]["id"] for doc in document_store._search_documents()}

    return docs.difference(file_ids | catalogue_ids)


def create_docs(files, meta, recreate_index: bool = False):
    unstructured_file_converter = UnstructuredFileConverter(
        api_key=os.getenv("UNSTRUCTURED_API_KEY")
    )
    cleaner = DocumentCleaner(
        remove_empty_lines=True,
        remove_extra_whitespaces=True,
        remove_repeated_substrings=True,
    )
    splitter = DocumentSplitter(split_by="passage", split_length=1, split_overlap=0)
    embedding = SentenceTransformersDocumentEmbedder(device="cuda")
    document_store = ElasticsearchDocumentStore(
        hosts="http://localhost:9200",
        basic_auth=(
            os.environ.get("ELASTIC_USERNAME"),
            os.environ.get("ELASTIC_PASSWORD"),
        ),
    )
    if recreate_index:
        from elasticsearch import Elasticsearch

        es = Elasticsearch(
            hosts="http://localhost:9200",
            basic_auth=(
                os.environ.get("ELASTIC_USERNAME"),
                os.environ.get("ELASTIC_PASSWORD"),
            ),
        )
        es.options(ignore_status=[400, 404]).indices.delete(index="default")

    index_pipeline = Pipeline(metadata=meta)
    index_pipeline.add_component("converter", unstructured_file_converter)
    index_pipeline.add_component("cleaner", cleaner)
    index_pipeline.add_component("splitter", splitter)
    index_pipeline.add_component("embedder", embedding)
    index_pipeline.add_component(
        "writer", DocumentWriter(document_store, policy=DuplicatePolicy.SKIP)
    )
    index_pipeline.connect("converter.documents", "cleaner.documents")
    index_pipeline.connect("cleaner.documents", "splitter.documents")
    index_pipeline.connect("splitter.documents", "embedder.documents")
    index_pipeline.connect("embedder.documents", "writer.documents")

    # _remove_old_docs(document_store)
    # files = _missing_docs(document_store)
    index_pipeline.run({"converter": {"paths": files}})


def query(query: str):
    document_store = ElasticsearchDocumentStore(
        hosts="http://localhost:9200",
        basic_auth=("elastic", "q0BWGooOpPwY3pTAUEmn"),
    )

    query_pipeline = Pipeline()
    query_pipeline.add_component(
        "text_embedder", SentenceTransformersTextEmbedder(device="cuda")
    )
    query_pipeline.add_component(
        "retriever", ElasticsearchEmbeddingRetriever(document_store=document_store)
    )
    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

    query = "housing"
    return query_pipeline.run({"text_embedder": {"text": query}})


def main():
    docs = list(Paths.DOCS_DIR.iterdir())
    docs_meta = [_add_metadata_to_document(doc.stem) for doc in docs]
    notes = list(Paths.NOTES_DIR.iterdir())
    notes_meta = [_add_metadata_to_document(note.stem) for note in notes]

    args.overwrite = True
    create_docs(docs, docs_meta, recreate_index=args.overwrite)
    create_docs(notes, notes_meta, recreate_index=False)  # never overwrite second run


if __name__ == "__main__":
    main()

    query("healthcare")["retriever"]["documents"][0]
