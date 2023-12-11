import json
import os
from argparse import ArgumentParser

from chroma_haystack import ChromaDocumentStore
from haystack import Document, Pipeline
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores import InMemoryDocumentStore
from unstructured_fileconverter_haystack import UnstructuredFileConverter

from src.common.utils import Paths, _add_metadata_to_document, validate_files

parser = ArgumentParser()
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite the existing index",
)
args = parser.parse_args()


files = list(Paths.DOCS_DIR.iterdir())


def create_docs(files, recreate_index: bool = False):
    unstructured_file_converter = UnstructuredFileConverter(
        api_key=os.getenv("UNSTRUCTURED_API_KEY")
    )
    splitter = DocumentSplitter(split_by="passage", split_length=1, split_overlap=0)
    document_store = InMemoryDocumentStore()

    indexing = Pipeline()
    indexing.add_component(
        instance=unstructured_file_converter,
        name="converter",
    )
    indexing.add_component("writer", DocumentWriter(document_store))
    indexing.connect("converter", "writer")

    indexing.run({"converter": {"paths": [str(file) for file in files]}})

    # _remove_old_docs(document_store)
    # missing = _missing_docs(document_store)
    #
    # for file in files:
    #     filename = file.stem[:6] if file.stem.startswith("notes-") else file.stem
    #     if filename not in missing and not recreate_index:
    #         continue
    #     metadata = _add_metadata_to_document(file.stem)
    #     indexing_pipeline.run(file_paths=file, meta=metadata)


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
