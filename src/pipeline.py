import json
import os
from argparse import ArgumentParser
from urllib.request import urlopen

import requests
from elasticsearch_haystack.document_store import ElasticsearchDocumentStore
from elasticsearch_haystack.embedding_retriever import (
    ElasticsearchEmbeddingRetriever
)
from haystack import Document, Pipeline, component
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder
)
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores import DuplicatePolicy, InMemoryDocumentStore
from tqdm import tqdm
from unstructured_fileconverter_haystack import UnstructuredFileConverter

from src.common.utils import (
    Paths,
    Urls,
    _add_metadata_to_document,
    validate_files
)


@component
class CDRCDocumentFetcher:
    def __init__(self, url: str = Urls.CATALOGUE):
        Paths.NOTES_DIR.mkdir(exist_ok=True, parents=True)
        Paths.DOCS_DIR.mkdir(exist_ok=True, parents=True)

        self.url = url

    @component.output_types(documents=list[Document])
    def run(self):
        self.get_metadata()
        self.get_files()
        self.create_docs()

    def get_metadata(self) -> list[dict]:
        response = urlopen(self.url)
        data = json.loads(response.read())["result"][0]

        self.metadata = []
        for item in tqdm(data):
            if "resources" not in item:
                continue
            for file in item["resources"]:
                if "profile" in file["name"].lower():
                    file["filename"] = file["url"].split("/")[-1]
                    file["parent_id"] = item["id"]
                    self.metadata.append(file)

            if "notes" not in item:
                continue
            out_file = Paths.NOTES_DIR / f"notes-{item['id']}.txt"
            if not out_file.exists():
                with open(out_file, "w") as f:
                    f.write(
                        f"Dataset Title: {item['title']} "
                        "\n\n Description: \n\n "
                        f"{re.sub('<[^<]+?>','', item['notes'])}"
                    )

        with open(Paths.DATA_DIR / "catalogue-metadata.json", "w") as f:
            json.dump(data, f)
        with open(Paths.DATA_DIR / "files-metadata.json", "w") as f:
            json.dump(self.metadata, f)

    def get_files(self) -> None:
        s = requests.Session()
        s.post(
            Urls.LOGIN,
            data={
                "name": os.getenv("name"),
                "pass": os.getenv("pass"),
                "form_build_id": os.getenv("form_build_id"),
                "form_id": "user_login",
                "op": "Log in",
            },
        )

        for meta in self.metadata:
            if meta["url"] != "":
                if (Paths.DOCS_DIR / meta["id"]).exists():
                    continue
                file = s.get(meta["url"])
                with open(Paths.DOCS_DIR / f"{meta['id']}.{meta['format']}", "wb") as f:
                    f.write(file.content)

    def create_docs(files, meta, recreate_index: bool = False):
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


tmp = CDRCDocumentFetcher().run()
