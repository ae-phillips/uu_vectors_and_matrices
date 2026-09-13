
import numpy as np
import pandas as pd

songs = pd.read_csv( "./data/song_vectors.csv.gz" )
vocab = pd.read_csv( "./data/vocabulary.csv" )
top_40 = pd.read_csv("./data/top_40_songs_by_genre.csv")

word_cols = vocab["word"].tolist() # Song count matrix
meta_cols = songs.columns.drop(word_cols).tolist() # Information about tracks

X = songs[word_cols].to_numpy(dtype=float)

# create new dataframe to save results to
genre_dot_results = top_40.copy()

# loop over top_40, match top_40 with its index position in songs and then use that for X matrix
for i in range(len(top_40)):
    song_id = top_40.iloc[i]["song_id"]
    song_index = songs[songs["song_id"] == song_id].index[0]
    song_vector = X[song_index]
    top_scores = X @ song_vector

    # Find the artist of the original song
    artist = songs.loc[song_index, "artist_name"]

    # Find all songs by that artist
    same_artist = songs["artist_name"] == artist

    # Remove all of them from consideration
    top_scores[same_artist] = -np.inf

    # Now find the top 3 remaining songs
    top_indices = np.argsort(top_scores)[::-1][:3]
    top_scores = top_scores[top_indices]

# create new columns for top 1,2,3 matches
    genre_dot_results.loc[i, "dot1_artist"] = songs.iloc[top_indices[0]]["artist_name"]
    genre_dot_results.loc[i, "dot1_song"] = songs.iloc[top_indices[0]]["song_title"]
    genre_dot_results.loc[i, "dot1_genre"] = songs.iloc[top_indices[0]]["genre"]
    genre_dot_results.loc[i, "dot1_score"] = top_scores[0]

    genre_dot_results.loc[i, "dot2_artist"] = songs.iloc[top_indices[1]]["artist_name"]
    genre_dot_results.loc[i, "dot2_song"] = songs.iloc[top_indices[1]]["song_title"]
    genre_dot_results.loc[i, "dot2_genre"] = songs.iloc[top_indices[1]]["genre"]
    genre_dot_results.loc[i, "dot2_score"] = top_scores[1]

    genre_dot_results.loc[i, "dot3_artist"] = songs.iloc[top_indices[2]]["artist_name"]
    genre_dot_results.loc[i, "dot3_song"] = songs.iloc[top_indices[2]]["song_title"]
    genre_dot_results.loc[i, "dot3_genre"] = songs.iloc[top_indices[2]]["genre"]
    genre_dot_results.loc[i, "dot3_score"] = top_scores[2]


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

genre_cos_results = top_40.copy()

for i in range(len(top_40)):
    song_id = top_40.iloc[i]["song_id"]
    song_index = songs[songs["song_id"] == song_id].index[0]
    song_vector = X[song_index]

    cos_top_scores = cosine_scores(X, song_vector)

    # Find the artist of the original song
    artist = songs.loc[song_index, "artist_name"]

    # Find all songs by that artist
    same_artist = songs["artist_name"] == artist

    # Remove all of them from consideration
    cos_top_scores[same_artist] = -np.inf

    # Now find the top 3 remaining songs
    cos_top_indices = np.argsort(cos_top_scores)[::-1][:3]
    cos_top_scores = cos_top_scores[cos_top_indices]


    genre_cos_results.loc[i, "cos1_artist"] = songs.iloc[cos_top_indices[0]]["artist_name"]
    genre_cos_results.loc[i, "cos1_song"] = songs.iloc[cos_top_indices[0]]["song_title"]
    genre_cos_results.loc[i, "cos1_genre"] = songs.iloc[cos_top_indices[0]]["genre"]
    genre_cos_results.loc[i, "cos1_score"] = cos_top_scores[0]

    genre_cos_results.loc[i, "cos2_artist"] = songs.iloc[cos_top_indices[1]]["artist_name"]
    genre_cos_results.loc[i, "cos2_song"] = songs.iloc[cos_top_indices[1]]["song_title"]
    genre_cos_results.loc[i, "cos2_genre"] = songs.iloc[cos_top_indices[1]]["genre"]
    genre_cos_results.loc[i, "cos2_score"] = cos_top_scores[1]

    genre_cos_results.loc[i, "cos3_artist"] = songs.iloc[cos_top_indices[2]]["artist_name"]
    genre_cos_results.loc[i, "cos3_song"] = songs.iloc[cos_top_indices[2]]["song_title"]
    genre_cos_results.loc[i, "cos3_genre"] = songs.iloc[cos_top_indices[2]]["genre"]
    genre_cos_results.loc[i, "cos3_score"] = cos_top_scores[2]


id_cols = ["track_id", "song_id", "artist_id", "artist_name", "song_title", "genre"]

genre_results = pd.concat(
    [
        genre_dot_results,
        genre_cos_results.drop(columns=id_cols)
    ],
    axis=1
)

# Term frequency: repeated uses of one word count less and less.
TF = np.log1p(X) # Compute TF

# Document frequency: how many songs use each word at least once.
N = X.shape[0] # Determine number of documents
df = (X > 0).sum(axis=0) # Compute document frequency
IDF = np.log(N / df) # Compute IDF

# IDF is one number per word, so NumPy broadcasts it across every song row.
TFIDF = TF * IDF


genre_cos_tfidf = top_40.copy()

for i in range(len(top_40)):
    song_id = top_40.iloc[i]["song_id"]
    song_index = songs[songs["song_id"] == song_id].index[0]
    song_vector = TFIDF[song_index]

    cos_tfidf_scores = cosine_scores(TFIDF, song_vector)

    # Find the artist of the original song
    artist = songs.loc[song_index, "artist_name"]

    # Find all songs by that artist
    same_artist = songs["artist_name"] == artist

    # Remove all of them from consideration
    cos_tfidf_scores[same_artist] = -np.inf

    # Now find the top 3 remaining songs
    cos_tfidf_indices = np.argsort(cos_tfidf_scores)[::-1][:3]
    cos_tfidf_scores = cos_tfidf_scores[cos_tfidf_indices]

    genre_cos_tfidf.loc[i, "tfidf_cos1_artist"] = songs.iloc[cos_tfidf_indices[0]]["artist_name"]
    genre_cos_tfidf.loc[i, "tfidf_cos1_song"] = songs.iloc[cos_tfidf_indices[0]]["song_title"]
    genre_cos_tfidf.loc[i, "tfidf_cos1_genre"] = songs.iloc[cos_tfidf_indices[0]]["genre"]
    genre_cos_tfidf.loc[i, "tfidf_cos1_score"] = cos_tfidf_scores[0]

    genre_cos_tfidf.loc[i, "tfidf_cos2_artist"] = songs.iloc[cos_tfidf_indices[1]]["artist_name"]
    genre_cos_tfidf.loc[i, "tfidf_cos2_song"] = songs.iloc[cos_tfidf_indices[1]]["song_title"]
    genre_cos_tfidf.loc[i, "tfidf_cos2_genre"] = songs.iloc[cos_tfidf_indices[1]]["genre"]
    genre_cos_tfidf.loc[i, "tfidf_cos2_score"] = cos_tfidf_scores[1]

    genre_cos_tfidf.loc[i, "tfidf_cos3_artist"] = songs.iloc[cos_tfidf_indices[2]]["artist_name"]
    genre_cos_tfidf.loc[i, "tfidf_cos3_song"] = songs.iloc[cos_tfidf_indices[2]]["song_title"]
    genre_cos_tfidf.loc[i, "tfidf_cos3_genre"] = songs.iloc[cos_tfidf_indices[2]]["genre"]
    genre_cos_tfidf.loc[i, "tfidf_cos3_score"] = cos_tfidf_scores[2]

genre_results = pd.concat(
    [
        genre_results,
        genre_cos_tfidf.drop(columns=id_cols)
    ],
    axis=1
)

