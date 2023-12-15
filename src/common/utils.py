import json
from pathlib import Path

from more_itertools import consume


class Paths:
    DATA_DIR = Path("./data")
    PROFILES_DIR = DATA_DIR / "profiles"
    DOCS_DIR = PROFILES_DIR / "docs"
    NOTES_DIR = PROFILES_DIR / "notes"


class Urls:
    CATALOGUE = (
        "https://data.cdrc.ac.uk/api/3/action/current_package_list_with_resources"
    )
    LOGIN = "https://data.cdrc.ac.uk/user/login"


def _add_metadata_to_document(doc_id: str) -> dict:
    with open(Paths.DATA_DIR / "catalogue-metadata.json") as f:
        catalogue_metadata = json.load(f)
    with open(Paths.DATA_DIR / "files-metadata.json") as f:
        files_metadata = json.load(f)

    if doc_id.startswith("notes-"):
        main_id = doc_id[6:]
    else:
        for file_meta in files_metadata:
            if doc_id == file_meta["id"]:
                main_id = file_meta["parent_id"]
                break

    for catalogue_meta in catalogue_metadata:
        if main_id == catalogue_meta["id"]:
            return {
                "title": catalogue_meta["title"],
                "notes": catalogue_meta["notes"],
                "id": catalogue_meta["id"],
                "url": catalogue_meta["url"],
            }


def validate_files(
    remove_old_files: bool = True,
) -> tuple[set[str], set[str], set[str], set[str]]:
    with open(Paths.DATA_DIR / "files-metadata.json", "r") as f:
        file_data = json.load(f)
    with open(Paths.DATA_DIR / "catalogue-metadata.json", "r") as f:
        catalogue_data = json.load(f)

    file_ids = {file["id"] for file in file_data}
    catalogue_ids = {catalogue["id"] for catalogue in catalogue_data}

    docs = list(Paths.DOCS_DIR.iterdir())
    notes = list(Paths.NOTES_DIR.iterdir())

    doc_ids = {doc.stem for doc in docs}
    note_ids = {note.stem[6:] for note in notes}

    docs_remove = doc_ids.difference(file_ids)
    notes_remove = note_ids.difference(catalogue_ids)

    if remove_old_files:
        consume(f.unlink() for f in docs if f.stem in docs_remove)
        consume(f.unlink() for f in notes if f.stem[6:] in notes_remove)

    docs_missing = file_ids.difference(doc_ids)
    notes_missing = catalogue_ids.difference(note_ids)

    return docs_remove, notes_remove, docs_missing, notes_missing
