import os

from haystack import Pipeline
from haystack.document_stores import PineconeDocumentStore
from haystack.nodes import (
    DocxToTextConverter,
    EmbeddingRetriever,
    FileTypeClassifier,
    PDFToTextConverter,
    PreProcessor,
    TextConverter
)

from src.common.utils import Paths, _add_metadata_to_document


def create_docs(files):
    document_store = PineconeDocumentStore(
        api_key=os.environ.get("PINECONE_API_KEY"),
        environment="gcp-starter",
        similarity="dot_product",
        embedding_dim=768,
        index="cdrc",
    )
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        split_by="passage",
        split_length=1,
        split_overlap=0,
        split_respect_sentence_boundary=False,
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

    for file in files:
        metadata = _add_metadata_to_document(file.stem)
        indexing_pipeline.run(file_paths=file, meta=metadata)


def main():
    files = list(Paths.DOCS_DIR.iterdir())
    create_docs(files)


if __name__ == "__main__":
    main()
