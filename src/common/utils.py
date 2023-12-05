import json
from pathlib import Path


class Paths:
    DATA_DIR = Path("./data")
    DOCS_DIR = DATA_DIR / Path("docs")
    NOTES_DIR = DATA_DIR / Path("notes")


def _add_metadata_to_document(doc_id):
    with open(Paths.DATA_DIR / "catalogue-metadata.json") as f:
        cmeta = json.load(f)
    with open(Paths.DATA_DIR / "files-metadata.json") as f:
        fmeta = json.load(f)

    if not doc_id.startswith("notes-"):
        for fm in fmeta:
            if doc_id == fm["id"]:
                parent_id = fm["parent_id"]

        for cm in cmeta:
            if parent_id == cm["id"]:
                return {"title": cm["title"], "notes": cm["notes"], "id": parent_id}
    else:
        for cm in cmeta:
            if doc_id.lstrip("notes-") == cm["id"]:
                return {"title": cm["title"], "notes": cm["notes"], "id": cm["id"]}
