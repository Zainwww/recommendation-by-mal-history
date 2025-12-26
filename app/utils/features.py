import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics.pairwise import cosine_similarity


# ===============================
# FEATURE BUILDING
# ===============================

def build_feature_matrix(dataset):
    # ---- MULTILABEL ----
    mlb_genre = MultiLabelBinarizer()
    mlb_studio = MultiLabelBinarizer()
    mlb_producer = MultiLabelBinarizer()

    X_genre = mlb_genre.fit_transform(dataset['genre'])
    X_studio = mlb_studio.fit_transform(dataset['studio'])
    X_producer = mlb_producer.fit_transform(dataset['producer'])

    # ---- YEAR (MINMAX) ----
    year_scaler = MinMaxScaler(feature_range=(0.0, 1.0))

    dataset['aired'] = pd.to_datetime(dataset['aired'], errors='coerce')
    dataset['year'] = dataset['aired'].dt.year
    dataset['year'] = pd.to_numeric(dataset['year'], errors='coerce')
    dataset['year'] = dataset['year'].fillna(dataset['year'].median())

    X_year = year_scaler.fit_transform(dataset[['year']])

    # ---- BERT ----
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')

    keywords_text = dataset['keywords'].apply(
        lambda x: ' '.join(x) if isinstance(x, list) else ''
    )

    bert_embeddings = model.encode(
        keywords_text.tolist(),
        batch_size=16,
        show_progress_bar=True
    )

    bert_embeddings = normalize(bert_embeddings)

    # ---- SPARSE ----
    X_genre = csr_matrix(X_genre, dtype=np.float32)
    X_studio = csr_matrix(X_studio, dtype=np.float32)
    X_producer = csr_matrix(X_producer, dtype=np.float32)
    X_bert = csr_matrix(bert_embeddings, dtype=np.float32)
    X_year = csr_matrix(X_year, dtype=np.float32)

    # ---- FINAL FEATURE MATRIX ----
    feature_matrix = hstack([
        X_genre * 2.5,
        X_studio * 2.0,
        X_producer * 1.75,
        X_bert * 1.0,
        X_year * 0.5
    ]).tocsr()

    return feature_matrix


# ===============================
# USER PROFILE
# ===============================

def build_user_profile(feature_matrix, liked_indices):
    """
    Average feature vector dari anime yang disukai user
    """
    if len(liked_indices) == 0:
        return None

    user_profile = feature_matrix[liked_indices].mean(axis=0)
    user_profile = normalize(user_profile)

    return user_profile
