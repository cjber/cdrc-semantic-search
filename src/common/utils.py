import json
import tomllib
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

with open("./config/config.toml", "rb") as f:
    Config = tomllib.load(f)


class CDRCSettings(BaseSettings):
    api_url: str
    login_url: str


class DataStoreSettings(BaseSettings):
    index_name: str = Field(min_length=1)
    hf_embed_dim: int = Field(gt=0, le=10_000)
    chunk_size: int = Field(gt=0, le=10_000)
    chunk_overlap: int = Field(ge=0, le=10_000)
    overwrite: bool


class ModelSettings(BaseSettings):
    top_k: int = Field(gt=0, le=20)
    vector_store_query_mode: str = Field(pattern="default|sparse|hybrid")
    alpha: float = Field(gt=0, le=1)
    prompt: str = Field(min_length=1)


class SharedSettings(BaseSettings):
    hf_embed_model: str = Field(min_length=1)


class Settings(BaseSettings):
    model: ModelSettings = ModelSettings.model_validate(Config["model"])
    datastore: DataStoreSettings = DataStoreSettings.model_validate(Config["datastore"])
    cdrc: CDRCSettings = CDRCSettings.model_validate(Config["cdrc-api"])
    shared: SharedSettings = SharedSettings.model_validate(Config["shared"])


class Paths:
    DATA_DIR: Path = Path("data")
    PROFILES_DIR: Path = DATA_DIR / "profiles"
    DOCS_DIR: Path = PROFILES_DIR / "docs"
    NOTES_DIR: Path = PROFILES_DIR / "notes"
    PIPELINE_STORAGE: Path = Path("./pipeline_storage")


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
