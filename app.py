import streamlit as st

from data_loader import load_resources
from search_engine import hybrid_search


@st.cache_resource
def initialize():

    texts, model, index, bm25 = (
        load_resources()
    )

    return texts, model, index, bm25


# load everything once
with st.spinner("Loading Search Engine..."):
    texts, model, index, bm25 = initialize()


# UI
st.set_page_config(
    page_title="Complaint Search",
    page_icon="🔍"
)

st.title("🔍 Hybrid Complaint Search")

st.write(
    "FAISS + SentenceTransformer + BM25"
)

query = st.text_input(
    "Enter Complaint Query"
)

top_k = st.slider(
    "Top Results",
    min_value=1,
    max_value=10,
    value=5
)


if st.button("Search"):

    if query.strip():

        results = hybrid_search(
            query,
            model,
            index,
            bm25,
            top_k
        )

        st.subheader("Results")

        for idx in results:

            st.markdown("---")

            st.write(
                f"📄 {texts[idx]}"
            )

    else:

        st.warning(
            "Please enter a query"
        )