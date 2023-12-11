from subprocess import Popen
from time import sleep

import polars as pl
import requests
import streamlit as st


def process_answer(r):
    df = pl.from_dict(
        {
            key: [value]
            for key, value in r.items()
            if key in ["answers", "documents", "explanation"]
        }
    ).explode(["answers", "documents", "explanation"])
    answers = (
        df["answers"].struct.unnest().with_columns(pl.col("offsets_in_context").list[0])
    )
    offsets = answers["offsets_in_context"].struct.unnest()
    documents = df["documents"].struct.unnest()
    document_meta = documents["meta"].struct.unnest()
    explanation = df["explanation"].to_frame()

    df = pl.concat(
        [
            answers[["context"]],
            offsets,
            document_meta[["title", "url"]],
            explanation,
        ],
        how="horizontal",
    )

    return (
        df.group_by(["title", "url"])
        .agg([pl.col("context"), pl.col("start"), pl.col("end"), pl.col("explanation")])
        .with_columns(pl.col("explanation").list.join("\n"))
    )


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

        df = process_answer(r)
        for row in df.rows(named=True):
            st.subheader(row["title"])
            st.write(
                f"_<span style ='color:lightgrey'>{row['explanation']}</span>_",
                unsafe_allow_html=True,
            )
            for context, start, end in zip(row["context"], row["start"], row["end"]):
                st.write(
                    f"...{context[:start]} **<span style ='color:lightblue'><u>{context[start:end]}</u></span>** {context[end:]}...",
                    unsafe_allow_html=True,
                )
            if row["url"] != "None":
                st.write(row["url"])
            st.write("----")


if __name__ == "__main__":
    main()
