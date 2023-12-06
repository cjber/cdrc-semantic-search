from subprocess import Popen

import polars as pl
import requests
import streamlit as st


def process_answer(r):
    titles = []
    contexts = []
    starts = []
    ends = []
    urls = []
    for answer, doc in zip(r["answers"], r["documents"]):
        offsets_in_context = answer["offsets_in_context"]
        starts.append(offsets_in_context[0]["start"])
        ends.append(offsets_in_context[0]["end"])
        titles.append(doc["meta"]["title"])
        contexts.append(answer["context"])
        if "url" in doc["meta"]:
            urls.append(doc["meta"]["url"])
        else:
            urls.append("None")

    return (
        pl.DataFrame(
            {
                "title": titles,
                "context": contexts,
                "start": starts,
                "end": ends,
                "url": urls,
            }
        )
        .group_by(["title", "url"])
        .agg([pl.col("context"), pl.col("start"), pl.col("end")])
    )


def main():
    Popen(["uvicorn", "search_service.api:app", "--port", "8000"])

    st.title("CDRC Semantic Search App")

    with st.spinner("Loading..."):
        while True:
            try:
                r = requests.get("http://localhost:8000/")
                if r.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                pass

    text = st.text_input("Query")
    while True:
        st.spinner("Waiting for input...")
        if text != "":
            break

    with st.spinner("Searching..."):
        r = requests.get("http://localhost:8000/query", params={"q": text})
        while r.status_code != 200:
            st.error("No results :(")
        r = r.json()

        df = process_answer(r)
        for row in df.rows(named=True):
            st.subheader(row["title"])
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
