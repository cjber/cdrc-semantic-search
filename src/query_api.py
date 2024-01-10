import json
import logging
import os
from dagster import asset
import re
from pathlib import Path
from urllib.request import urlopen

import requests
from tqdm import tqdm

from src.common.utils import Paths, Settings


CDRC = Settings().cdrc


@asset(group_name="cdrc", compute_kind="CDRC API")
def get_metadata() -> list[dict]:
    response = urlopen(CDRC.api_url)
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
        out_file = Paths.PROFILES_DIR / f"notes-{item['id']}.txt"
        if out_file.exists():
            logging.info(f"Skipping {out_file} as it already exists")
            continue
        with open(out_file, "w") as f:
            f.write(
                f"Dataset Title: {item['title']} "
                "\n\n Description: \n\n "
                f"{re.sub('<[^<]+?>','', item['notes'])}"
            )


login_details = {
    "name": os.getenv("name"),
    "pass": os.getenv("pass"),
    "form_build_id": os.getenv("form_build_id"),
}


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
        filename = (
            f"profile-{meta['id']}.{meta['format']}"
            if "profile" in meta["name"].lower()
            else f"flyer-{meta['id']}.{meta['format']}"
        )
        if (meta["url"] == "") or (self.profiles_dir / filename).exists():
            logging.info(f"Skipping {filename} as it already exists")
            continue
        file = s.get(meta["url"])
        with open(self.profiles_dir / filename, "wb") as f:
            f.write(file.content)


#
#     def write_metadata(self) -> None:
#         with open(self.data_dir / "catalogue-metadata.json", "w") as f:
#             json.dump(self.catalogue_meta, f)
#         with open(self.data_dir / "files-metadata.json", "w") as f:
#             json.dump(self.files_meta, f)
#
#
# def pipeline():
#     self.files_meta, self.catalogue_meta = self.get_metadata()
#     self.file_ids = {file["id"] for file in self.files_meta}
#     self.catalogue_ids = {catalogue["id"] for catalogue in self.catalogue_meta}
#
#     self.write_metadata()
#     self.download_files()
#
#
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.ERROR, filename="logs/query_api.log", filemode="w"
    )

    query = CDRCQuery(**Settings().cdrc.model_dump())
    query.run()
