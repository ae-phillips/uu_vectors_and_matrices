
## CHAT-GPT helped adapt this code to include a jaccard similarity calculation
## Comparing tags was my original idea, but CHAT-GPT gave a suggestion for easiest implementation
# it also edited this code to make it more efficient as compared to my previous attempts

import numpy as np
import pandas as pd


def cosine_scores(matrix, query_vector):
    numerators = matrix @ query_vector
    matrix_norms = np.linalg.norm(matrix, axis=1)
    query_norm = np.linalg.norm(query_vector)
    denominators = matrix_norms * query_norm

    return np.divide(
        numerators,
        denominators,
        out=np.zeros_like(numerators),
        where=denominators != 0,
    )


def run_tag_tfidf(
    songs,
    top_40,
    TFIDF,
    songs_tags,
    tag_jaccard,
    cosine_weight=0.8,
    tag_weight=0.2,
):

    tag_tfidf = top_40.copy()

    for i in range(len(top_40)):

        song_id = top_40.iloc[i]["song_id"]
        song_index = songs[songs["song_id"] == song_id].index[0]

        # TF-IDF cosine
        song_vector = TFIDF[song_index]
        cos_scores = cosine_scores(TFIDF, song_vector)

        # Jaccard tag similarity
        query_tags = songs_tags.loc[song_index, "clean_tags"]

        tag_scores = np.array([
            tag_jaccard(
                query_tags,
                songs_tags.loc[j, "clean_tags"]
            )
            for j in range(len(songs_tags))
        ])

        # Combine scores
        combined_scores = (
            cosine_weight * cos_scores
            + tag_weight * tag_scores
        )

        # Remove query artist
        artist = songs.loc[song_index, "artist_name"]
        same_artist = songs["artist_name"] == artist
        combined_scores[same_artist] = -np.inf

        # Top 3
        top_indices = np.argsort(combined_scores)[::-1][:3]
        top_scores = combined_scores[top_indices]

        for rank in range(3):
            match_index = top_indices[rank]

            tag_tfidf.loc[i, f"match{rank + 1}_artist"] = (
                songs.iloc[match_index]["artist_name"]
            )

            tag_tfidf.loc[i, f"match{rank + 1}_song"] = (
                songs.iloc[match_index]["song_title"]
            )

            tag_tfidf.loc[i, f"match{rank + 1}_genre"] = (
                songs.iloc[match_index]["genre"]
            )

            tag_tfidf.loc[i, f"match{rank + 1}_score"] = (
                top_scores[rank]
            )

    return tag_tfidf