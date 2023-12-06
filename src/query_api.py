import json
import os
import re
from urllib.request import urlopen

import requests

from src.common.utils import Paths


def get_docs() -> list[dict]:
    response = urlopen(
        "https://data.cdrc.ac.uk/api/3/action/current_package_list_with_resources"
    )
    data = json.loads(response.read())["result"][0]

    docs = []
    for item in data:
        if "resources" not in item:
            continue
        for file in item["resources"]:
            if "profile" in file["name"].lower():
                file["filename"] = file["url"].split("/")[-1]
                file["parent_id"] = item["id"]
                docs.append(file)

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
        json.dump(docs, f)

    return docs


def get_files(docs: list[dict]) -> None:
    s = requests.Session()
    s.post(
        "https://data.cdrc.ac.uk/user/login",
        data={
            "name": os.getenv("name"),
            "pass": os.getenv("pass"),
            "form_build_id": os.getenv("form_build_id"),
            "form_id": "user_login",
            "op": "Log in",
        },
    )

    for doc in docs:
        if doc["url"] != "":
            if (Paths.DOCS_DIR / doc["id"]).exists():
                continue
            file = s.get(doc["url"])
            with open(Paths.DOCS_DIR / f"{doc['id']}.{doc['format']}", "wb") as f:
                f.write(file.content)


def main():
    Paths.NOTES_DIR.mkdir(exist_ok=True, parents=True)
    Paths.DOCS_DIR.mkdir(exist_ok=True, parents=True)

    docs = get_docs()
    get_files(docs)


if __name__ == "__main__":
    main()
