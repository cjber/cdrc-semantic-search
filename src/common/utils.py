import json
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Paths:
    DATA_DIR = Path("data")
    PROFILES_DIR = DATA_DIR / "profiles"
    DOCS_DIR = PROFILES_DIR / "docs"
    NOTES_DIR = PROFILES_DIR / "notes"
    PIPELINE_STORAGE = Path("./pipeline_storage")


class Consts:
    INDEX_NAME = "cdrc-index"
    HF_EMBED_MODEL = None
    EMBED_DIM = 384
    PROMPT = (
        "Below is the data profile for a dataset from the CDRC Data Catalogue.\n"
        "1. Summarise the data profile in under 50 words.\n"
        "2. Explain the relevance of the dataset to the following query, using your own knowledge or the documents.\n\n"
        "For each unique dataset, structure your answer as follows:\n\n"
        "Title: <dataset title>\n\n"
        "Summary: <dataset summary>\n\n"
        "Relevance: <dataset relevance to 'Query'>\n\n"
        "\n---------------------\n"
        "Query: {query_str}\n\n"
        "Data profile:\n"
        "\n---------------------\n{context_str}\n---------------------\n\n"
    )


class Urls:
    API_URL = "https://data.cdrc.ac.uk/api/3/action/current_package_list_with_resources"
    LOGIN_URL = "https://data.cdrc.ac.uk/user/login"


class ModelSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="model_")
    top_k: int = Field(5)
    llm: bool = Field(True)
    vector_store_query_mode: str = Field("hybrid", pattern="default|sparse|hybrid")
    alpha: float = Field(0.5)


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
            }
