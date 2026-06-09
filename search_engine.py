import faiss
import numpy as np


def hybrid_search(
    query,
    model,
    index,
    bm25,
    k=5
):

    # Convert query to embedding
    q_vec = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    # Normalize query
    faiss.normalize_L2(q_vec)

    # FAISS search
    _, faiss_indexes = index.search(
        q_vec,
        k
    )

    # BM25 search
    bm25_scores = bm25.get_scores(
        query.lower().split()
    )

    bm25_indexes = np.argsort(
        bm25_scores
    )[::-1][:k]

    # Combine both searches
    combined_indexes = (
        list(faiss_indexes[0])
        + list(bm25_indexes)
    )

    # Remove duplicates
    final_indexes = list(
        dict.fromkeys(combined_indexes)
    )

    return final_indexes[:k]