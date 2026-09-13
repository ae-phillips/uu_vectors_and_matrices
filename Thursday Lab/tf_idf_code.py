
import numpy as np
import pandas as pd

songs = pd.read_csv( "./data/song_vectors.csv.gz" )
vocab = pd.read_csv( "./data/vocabulary.csv" )
top_40 = pd.read_csv("./data/top_40_songs_by_genre.csv")

word_cols = vocab["word"].tolist() # Song count matrix
meta_cols = songs.columns.drop(word_cols).tolist() # Information about tracks


X = songs[word_cols].to_numpy(dtype=float)


# Term frequency: repeated uses of one word count less and less.
TF = np.log1p(X) # Compute TF

# Document frequency: how many songs use each word at least once.
N = X.shape[0] # Determine number of documents
df = (X > 0).sum(axis=0) # Compute document frequency
IDF = np.log(N / df) # Compute IDF

# IDF is one number per word, so NumPy broadcasts it across every song row.
TFIDF = TF * IDF


# create new dataframe to save results to
dot_top_tfidf = top_40.copy()

# loop over top_40, match top_40 with its index position in songs and then use that for X matrix
for i in range(len(top_40)):
    song_id = top_40.iloc[i]["song_id"]
    song_index = songs[songs["song_id"] == song_id].index[0]
    song_vector = TFIDF[song_index]
    top_scores = TFIDF @ song_vector
    top_scores[song_index] = -np.inf
    top_indices = np.argsort(top_scores)[::-1][:3]
    top_scores = top_scores[top_indices]

# create new columns for top 1,2,3 matches
    dot_top_tfidf.loc[i, "dot1_artist"] = songs.iloc[top_indices[0]]["artist_name"]
    dot_top_tfidf.loc[i, "dot1_song"] = songs.iloc[top_indices[0]]["song_title"]
    dot_top_tfidf.loc[i, "dot1_genre"] = songs.iloc[top_indices[0]]["genre"]
    dot_top_tfidf.loc[i, "dot1_score"] = top_scores[0]

    dot_top_tfidf.loc[i, "dot2_artist"] = songs.iloc[top_indices[1]]["artist_name"]
    dot_top_tfidf.loc[i, "dot2_song"] = songs.iloc[top_indices[1]]["song_title"]
    dot_top_tfidf.loc[i, "dot2_genre"] = songs.iloc[top_indices[1]]["genre"]
    dot_top_tfidf.loc[i, "dot2_score"] = top_scores[1]

    dot_top_tfidf.loc[i, "dot3_artist"] = songs.iloc[top_indices[2]]["artist_name"]
    dot_top_tfidf.loc[i, "dot3_song"] = songs.iloc[top_indices[2]]["song_title"]
    dot_top_tfidf.loc[i, "dot3_genre"] = songs.iloc[top_indices[2]]["genre"]
    dot_top_tfidf.loc[i, "dot3_score"] = top_scores[2]




# make another dataframe copy for centered inner product results
centered_top_tfidf = top_40.copy()

for i in range(len(top_40)):
    song_id = top_40.iloc[i]["song_id"]
    song_index = songs[songs["song_id"] == song_id].index[0]
    song_vector = TFIDF[song_index]
    song_centered = song_vector - song_vector.mean()
    X_centered = TFIDF - TFIDF.mean(axis=1, keepdims=True)
    centered_top_scores = X_centered @ song_centered
    centered_top_scores[song_index] = -np.inf
    centered_top_indices = np.argsort(centered_top_scores)[::-1][:3]
    centered_top_scores = centered_top_scores[centered_top_indices]

    centered_top_tfidf.loc[i, "centered1_artist"] = songs.iloc[centered_top_indices[0]]["artist_name"]
    centered_top_tfidf.loc[i, "centered1_song"] = songs.iloc[centered_top_indices[0]]["song_title"]
    centered_top_tfidf.loc[i, "centered1_genre"] = songs.iloc[centered_top_indices[0]]["genre"]
    centered_top_tfidf.loc[i, "centered1_score"] = centered_top_scores[0]

    centered_top_tfidf.loc[i, "centered2_artist"] = songs.iloc[centered_top_indices[1]]["artist_name"]
    centered_top_tfidf.loc[i, "centered2_song"] = songs.iloc[centered_top_indices[1]]["song_title"]
    centered_top_tfidf.loc[i, "centered2_genre"] = songs.iloc[centered_top_indices[1]]["genre"]
    centered_top_tfidf.loc[i, "centered2_score"] = centered_top_scores[1]

    centered_top_tfidf.loc[i, "centered3_artist"] = songs.iloc[centered_top_indices[2]]["artist_name"]
    centered_top_tfidf.loc[i, "centered3_song"] = songs.iloc[centered_top_indices[2]]["song_title"]
    centered_top_tfidf.loc[i, "centered3_genre"] = songs.iloc[centered_top_indices[2]]["genre"]
    centered_top_tfidf.loc[i, "centered3_score"] = centered_top_scores[2]



# create id column name list for easier concatenation
id_cols = ["track_id", "song_id", "artist_id", "artist_name", "song_title", "genre"]

top_40_tf_idf = pd.concat(
    [
        dot_top_tfidf,
        centered_top_tfidf.drop(columns=id_cols)
    ],
    axis=1
)

corr_top_tfidf = top_40.copy()

for i in range(len(top_40)):
    song_id = top_40.iloc[i]["song_id"]
    song_index = songs[songs["song_id"] == song_id].index[0]
    song_vector = TFIDF[song_index]

    song_centered = song_vector - song_vector.mean()
    X_centered = TFIDF - TFIDF.mean(axis=1, keepdims=True)
    centered_top_scores = X_centered @ song_centered

    top_covariance_scores = centered_top_scores / TFIDF.shape[1]
    song_std = song_vector.std()
    X_std = TFIDF.std(axis=1)
    denom = X_std * song_std

    corr_top_scores = np.divide(
    top_covariance_scores,
    denom,
    out=np.zeros_like(top_covariance_scores),
    where=denom != 0,)


    corr_top_scores[song_index] = -np.inf
    corr_top_indices = np.argsort(corr_top_scores)[::-1][:3]
    corr_top_scores = corr_top_scores[corr_top_indices]

    corr_top_tfidf.loc[i, "corr1_artist"] = songs.iloc[corr_top_indices[0]]["artist_name"]
    corr_top_tfidf.loc[i, "corr1_song"] = songs.iloc[corr_top_indices[0]]["song_title"]
    corr_top_tfidf.loc[i, "corr1_genre"] = songs.iloc[corr_top_indices[0]]["genre"]
    corr_top_tfidf.loc[i, "corr1_score"] = corr_top_scores[0]

    corr_top_tfidf.loc[i, "corr2_artist"] = songs.iloc[corr_top_indices[1]]["artist_name"]
    corr_top_tfidf.loc[i, "corr2_song"] = songs.iloc[corr_top_indices[1]]["song_title"]
    corr_top_tfidf.loc[i, "corr2_genre"] = songs.iloc[corr_top_indices[1]]["genre"]
    corr_top_tfidf.loc[i, "corr2_score"] = corr_top_scores[1]

    corr_top_tfidf.loc[i, "corr3_artist"] = songs.iloc[corr_top_indices[2]]["artist_name"]
    corr_top_tfidf.loc[i, "corr3_song"] = songs.iloc[corr_top_indices[2]]["song_title"]
    corr_top_tfidf.loc[i, "corr3_genre"] = songs.iloc[corr_top_indices[2]]["genre"]
    corr_top_tfidf.loc[i, "corr3_score"] = corr_top_scores[2]

top_40_tf_idf = pd.concat(
    [
        top_40_tf_idf,
        corr_top_tfidf.drop(columns=id_cols)
    ],
    axis=1
)

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

cos_top_tfidf = top_40.copy()

for i in range(len(top_40)):
    song_id = top_40.iloc[i]["song_id"]
    song_index = songs[songs["song_id"] == song_id].index[0]
    song_vector = TFIDF[song_index]

    cos_top_scores = cosine_scores(TFIDF, song_vector)

    cos_top_scores[song_index] = -np.inf
    cos_top_indices = np.argsort(cos_top_scores)[::-1][:3]
    cos_top_scores = cos_top_scores[cos_top_indices]

    cos_top_tfidf.loc[i, "cos1_artist"] = songs.iloc[cos_top_indices[0]]["artist_name"]
    cos_top_tfidf.loc[i, "cos1_song"] = songs.iloc[cos_top_indices[0]]["song_title"]
    cos_top_tfidf.loc[i, "cos1_genre"] = songs.iloc[cos_top_indices[0]]["genre"]
    cos_top_tfidf.loc[i, "cos1_score"] = cos_top_scores[0]

    cos_top_tfidf.loc[i, "cos2_artist"] = songs.iloc[cos_top_indices[1]]["artist_name"]
    cos_top_tfidf.loc[i, "cos2_song"] = songs.iloc[cos_top_indices[1]]["song_title"]
    cos_top_tfidf.loc[i, "cos2_genre"] = songs.iloc[cos_top_indices[1]]["genre"]
    cos_top_tfidf.loc[i, "cos2_score"] = cos_top_scores[1]

    cos_top_tfidf.loc[i, "cos3_artist"] = songs.iloc[cos_top_indices[2]]["artist_name"]
    cos_top_tfidf.loc[i, "cos3_song"] = songs.iloc[cos_top_indices[2]]["song_title"]
    cos_top_tfidf.loc[i, "cos3_genre"] = songs.iloc[cos_top_indices[2]]["genre"]
    cos_top_tfidf.loc[i, "cos3_score"] = cos_top_scores[2]

top_40_tf_idf = pd.concat(
    [
        top_40_tf_idf,
        cos_top_tfidf.drop(columns=id_cols)
    ],
    axis=1
)

