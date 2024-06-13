from subprocess import Popen
from time import sleep

import requests
import streamlit as st


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
                sleep(10)

    # use_llm = st.toggle("Activate LLM")
    use_llm = False
    text = st.text_input("Query")
    if text == "":
        return None

    r = requests.get(
        "http://localhost:8000/query", params={"q": text, "use_llm": use_llm}
    )
    if r.status_code != 200:
        st.error("No results :(")
        return None

    if use_llm:
        response, metadata = r.json()
        responses = []
        for r in response["response"].split("---------------------"):
            if all(x in r for x in ["Summary: ", "Relevance: "]):
                responses.append(r)
            else:
                responses.append(None)

        for res, meta in zip(responses, metadata.values(), strict=False):
            st.subheader(meta["title"])
            if res:
                summary, relevance = res.split("Summary: ")[1].split("Relevance: ")
                st.caption(summary)
                st.caption(f":red[{relevance}]")
            # st.caption(f"Score: :red[{meta['score']:.3f}]")
            else:
                st.caption("LLM did not return a response.")
            if meta["url"] != "None":
                st.write(meta["url"])
            st.divider()
    else:
        metadata = r.json()
        for meta in metadata:
            st.subheader(meta["title"])
            st.caption(f"Score: :red[{meta['score']:.3f}]")
            if meta["url"] != "None":
                st.write(meta["url"])
            st.divider()


if __name__ == "__main__":
    main()
