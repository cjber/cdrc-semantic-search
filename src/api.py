import json
import os
from urllib.request import urlopen

import requests

import src.params as params


def get_docs() -> list[dict]:
    response = urlopen(
        "https://data.cdrc.ac.uk/api/3/action/current_package_list_with_resources"
    )
    data = json.loads(response.read())["result"][0]

    params.DATA_DIR.mkdir(exist_ok=True, parents=True)
    with open(params.DATA_DIR / "catalogue-metadata.json", "w") as f:
        json.dump(data, f)

    docs = []
    for item in data:
        if "resources" not in item:
            continue
        for file in item["resources"]:
            if "profile" in file["name"].lower():
                file["filename"] = file["url"].split("/")[-1]
                file["parent_id"] = item["id"]
                docs.append(file)

    with open(params.DATA_DIR / "files-metadata.json", "w") as f:
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
    params.PDF_DIR.mkdir(exist_ok=True, parents=True)

    for doc in docs:
        if doc["url"] != "":
            if (params.PDF_DIR / doc["id"]).exists():
                continue
            file = s.get(doc["url"])
            with open(params.PDF_DIR / f"{doc["id"]}.{doc["format"]}", "wb") as f:
                f.write(file.content)


def main():
    docs = get_docs()
    get_files(docs)


if __name__ == "__main__":
    main()
