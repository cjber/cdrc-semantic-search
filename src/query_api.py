import json
import logging
import os
import re
from pathlib import Path
from urllib.request import urlopen

import requests
from more_itertools import consume
from tqdm import tqdm

from src.common.utils import Paths, Urls

logging.getLogger().setLevel(logging.INFO)


class CDRCQuery:
    def __init__(
        self,
        api_url: str = Urls.API_URL,
        login_url: str = Urls.LOGIN_URL,
        data_dir: Path = Paths.DATA_DIR,
        profiles_dir: Path = Paths.PROFILES_DIR,
        login_details: dict = {
            "name": os.getenv("name"),
            "pass": os.getenv("pass"),
            "form_build_id": os.getenv("form_build_id"),
        },
        _remove_old_files: bool = True,
    ):
        self.api_url = api_url
        self.login_url = login_url
        self.data_dir = data_dir
        self.profiles_dir = profiles_dir
        self.login_details = login_details
        self._remove_old_files = _remove_old_files

        self.profiles_dir.mkdir(exist_ok=True, parents=True)

    def run(self):
        self.files_meta, self.catalogue_meta = self.get_metadata()
        self.write_metadata()
        self.download_files()

        if self._remove_old_files:
            self.remove_old_files()

    def get_metadata(self) -> list[dict]:
        response = urlopen(self.api_url)
        catalogue_meta = json.loads(response.read())["result"][0]

        files_meta = []
        for item in catalogue_meta:
            if "resources" not in item:
                continue
            for file in item["resources"]:
                if any(x in file["name"].lower() for x in ["profile", "flyer"]):
                    file["filename"] = file["url"].split("/")[-1]
                    file["parent_id"] = item["id"]
                    files_meta.append(file)

            if "notes" not in item:
                continue
            out_file = self.profiles_dir / f"notes-{item['id']}.txt"
            if out_file.exists():
                logging.info(f"Skipping {out_file} as it already exists")
                continue
            with open(out_file, "w") as f:
                f.write(
                    f"Dataset Title: {item['title']} "
                    "\n\n Description: \n\n "
                    f"{re.sub('<[^<]+?>','', item['notes'])}"
                )

        return files_meta, catalogue_meta

    def download_files(self) -> None:
        s = requests.Session()
        s.post(
            self.login_url,
            data={
                **self.login_details,
                "form_id": "user_login",
                "op": "Log in",
            },
        )

        for meta in tqdm(self.files_meta, desc="Downloading files"):
            filename = f"{meta['id']}.{meta['format']}"
            if (meta["url"] == "") or (self.profiles_dir / filename).exists():
                logging.info(f"Skipping {filename} as it already exists")
                continue
            file = s.get(meta["url"])
            with open(self.profiles_dir / filename, "wb") as f:
                f.write(file.content)

    def write_metadata(self) -> None:
        with open(self.data_dir / "catalogue-metadata.json", "w") as f:
            json.dump(self.catalogue_meta, f)
        with open(self.data_dir / "files-metadata.json", "w") as f:
            json.dump(self.files_meta, f)

    def remove_old_files(self):
        file_ids = {file["id"] for file in self.files_meta}
        catalogue_ids = {catalogue["id"] for catalogue in self.catalogue_meta}

        docs = [
            file
            for file in self.profiles_dir.iterdir()
            if not file.stem.startswith("notes-")
        ]
        notes = [
            file
            for file in self.profiles_dir.iterdir()
            if file.stem.startswith("notes-")
        ]
        doc_ids = {doc.stem for doc in docs}
        note_ids = {note.stem[6:] for note in notes}

        docs_remove = doc_ids.difference(file_ids)
        notes_remove = note_ids.difference(catalogue_ids)

        logging.info(f"Removing {len(docs_remove)} old documents")
        consume(f.unlink() for f in docs if f.stem in docs_remove)

        logging.info(f"Removing {len(notes_remove)} old notes")
        consume(f.unlink() for f in notes if f.stem[6:] in notes_remove)


if __name__ == "__main__":
    query = CDRCQuery()
    query.run()
