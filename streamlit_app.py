from subprocess import Popen
from time import sleep

import polars as pl
import requests
import streamlit as st

from src.common.utils import _add_metadata_to_document


def main():
    st.title("CDRC Semantic Search App")

    with st.spinner("Loading..."):
        while True:
            try:
                r = requests.get("http://localhost:8000/")
                if r.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                Popen(["uvicorn", "search_service.api:app", "--port", "8000"])
                sleep(5)

    text = st.text_input("Query")
    if text == "":
        return None

    with st.spinner("Searching..."):
        r = requests.get("http://localhost:8000/query", params={"q": text})
        if r.status_code != 200:
            st.error("No results :(")
            return None
        r = r.json()

        metadata = [
            _add_metadata_to_document(item["file_name"].split(".")[0])
            for key, item in r["metadata"].items()
        ]
        df = pl.DataFrame(metadata)

        st.write(f"{r['response']}")
        st.write("----")
        for row in df.rows(named=True):
            st.subheader(row["title"])
            if row["url"] != "None":
                st.write(row["url"])
            st.write("----")


if __name__ == "__main__":
    main()
