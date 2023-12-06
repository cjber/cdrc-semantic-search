import json
import os
from argparse import ArgumentParser

from haystack import Pipeline
from haystack.document_stores import PineconeDocumentStore
from haystack.nodes import (
    DocxToTextConverter,
    EmbeddingRetriever,
    FileTypeClassifier,
    PDFToTextConverter,
    PreProcessor,
    TextConverter,
)

from src.common.utils import Paths, _add_metadata_to_document, validate_files

parser = ArgumentParser()
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite the existing index",
)
args = parser.parse_args()


def create_docs(files, recreate_index: bool = False):
    document_store = PineconeDocumentStore(
        api_key=os.environ.get("PINECONE_API_KEY"),
        environment="gcp-starter",
        similarity="dot_product",
        embedding_dim=768,
        index="cdrc",
        recreate_index=recreate_index,
    )
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        split_by="word",
        split_length=100,
        split_overlap=10,
        split_respect_sentence_boundary=True,
    )
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    )

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_node(
        FileTypeClassifier(), name="FileTypeClassifier", inputs=["File"]
    )
    indexing_pipeline.add_node(
        component=TextConverter(remove_numeric_tables=True, valid_languages=["en"]),
        name="TextConverter",
        inputs=["FileTypeClassifier.output_1"],
    )
    indexing_pipeline.add_node(
        component=PDFToTextConverter(
            remove_numeric_tables=True, valid_languages=["en"]
        ),
        name="PDFConverter",
        inputs=["FileTypeClassifier.output_2"],
    )
    indexing_pipeline.add_node(
        component=DocxToTextConverter(valid_languages=["en"]),
        name="DocxConverter",
        inputs=["FileTypeClassifier.output_4"],
    )
    indexing_pipeline.add_node(
        component=preprocessor,
        name="PreProcessor",
        inputs=["TextConverter", "PDFConverter", "DocxConverter"],
    )
    indexing_pipeline.add_node(
        component=retriever,
        name="Retriever",
        inputs=["PreProcessor"],
    )
    indexing_pipeline.add_node(
        component=document_store,
        name="DocumentStore",
        inputs=["Retriever"],
    )

    _remove_old_docs(document_store)
    missing = _missing_docs(document_store)

    for file in files:
        filename = file.stem[:6] if file.stem.startswith("notes-") else file.stem
        if filename not in missing and not recreate_index:
            continue
        metadata = _add_metadata_to_document(file.stem)
        indexing_pipeline.run(file_paths=file, meta=metadata)


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
    docs = {doc.to_dict()["meta"]["id"] for doc in document_store.get_all_documents()}

    return docs.difference(file_ids | catalogue_ids)


def main():
    docs = list(Paths.DOCS_DIR.iterdir())
    notes = list(Paths.NOTES_DIR.iterdir())

    create_docs(docs, recreate_index=args.overwrite)
    create_docs(notes, recreate_index=False)  # never overwrite second run


if __name__ == "__main__":
    main()
