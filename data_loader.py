import pandas as pd
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


def load_resources():

    
    df = pd.read_csv("complaints.csv")

    texts = df["complaint"].tolist()

    # Load model
    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    # create embeddings
    embeddings = model.encode(
        texts,
        convert_to_numpy=True
    ).astype("float32")

    # normalize embeddings
    faiss.normalize_L2(embeddings)

    # create FAISS index
    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    # create BM25 index
    tokenized = [
        text.lower().split()
        for text in texts
    ]

    bm25 = BM25Okapi(tokenized)

    return texts, model, index, bm25