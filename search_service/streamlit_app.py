from subprocess import Popen

import requests
import streamlit as st


def main():
    Popen(["uvicorn", "search_service.api:app", "--port", "8000"])

    st.title("CDRC Semantic Search App")
    text = st.text_input("Query")
    try:
        r = requests.get("http://localhost:8000/query", params={"q": text})
        r = r.json()
    except requests.exceptions.ConnectionError:
        r = None

    if r:
        for item in zip(r["answers"], r["documents"]):
            offsets_in_context = item[0]["offsets_in_context"]
            start = offsets_in_context[0]["start"]
            end = offsets_in_context[0]["end"]
            context = item[0]["context"]
            st.write(f"**{item[1]['meta']['title']}**")
            st.write(f"{context[:start]} **{context[start:end]}** {context[end:]}")
            # st.write(item[1]["meta"]["url"]) # TODO: add url to metadata
            st.write("----")


if __name__ == "__main__":
    main()
