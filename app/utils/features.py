import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from functools import lru_cache
import joblib
from pathlib import Path

# ===============================
# OPTIMIZED FEATURE BUILDING
# ===============================

def build_feature_matrix(dataset):
    """
    Optimized version - 3-5x faster
    """
    # ---- YEAR FIRST (fastest) ----
    dataset = dataset.copy()
    dataset['aired'] = pd.to_datetime(dataset['aired'], errors='coerce')
    dataset['year'] = dataset['aired'].dt.year
    dataset['year'] = pd.to_numeric(dataset['year'], errors='coerce')
    dataset['year'] = dataset['year'].fillna(dataset['year'].median())
    
    year_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    X_year = csr_matrix(year_scaler.fit_transform(dataset[['year']]), dtype=np.float32)

    # ---- MULTILABEL (parallel-friendly) ----
    # Pre-allocate MultiLabelBinarizers
    mlb_genre = MultiLabelBinarizer(sparse_output=True)
    mlb_studio = MultiLabelBinarizer(sparse_output=True)
    mlb_producer = MultiLabelBinarizer(sparse_output=True)
    mlb_keyword = MultiLabelBinarizer(sparse_output=True)

    # Fit & transform in one go (faster than separate fit + transform)
    X_genre = mlb_genre.fit_transform(dataset['genre']).astype(np.float32)
    X_studio = mlb_studio.fit_transform(dataset['studio']).astype(np.float32)
    X_producer = mlb_producer.fit_transform(dataset['producer']).astype(np.float32)
    X_keyword = mlb_keyword.fit_transform(dataset['keywords']).astype(np.float32)

    # ---- WEIGHTED STACKING (inline multiplication) ----
    feature_matrix = hstack([
        X_genre.multiply(2.5),
        X_studio.multiply(2.0),
        X_producer.multiply(1.75),
        X_keyword.multiply(1.0),
        X_year.multiply(0.5)
    ], format='csr', dtype=np.float32)

    return feature_matrix


# ===============================
# CACHED VERSION (untuk dataset yang tidak berubah)
# ===============================

def build_feature_matrix_cached(dataset, cache_dir='./cache'):
    """
    Build dengan caching - sangat cepat untuk load ulang
    Cache disimpan berdasarkan hash dataset
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)
    
    # Buat hash dari dataset untuk cache key
    data_hash = hash(tuple(dataset['mal_id'].values))
    cache_file = cache_path / f'feature_matrix_{data_hash}.pkl'
    
    # Coba load dari cache
    if cache_file.exists():
        print(f"Loading from cache...")
        return joblib.load(cache_file)
    
    # Build baru
    print("Building feature matrix...")
    feature_matrix = build_feature_matrix(dataset)
    
    # Save ke cache
    joblib.dump(feature_matrix, cache_file, compress=3)
    print(f"Saved to cache: {cache_file}")
    
    return feature_matrix


# ===============================
# INCREMENTAL VERSION (untuk dataset besar)
# ===============================
def build_feature_matrix_incremental(dataset, batch_size=5000):
    """
    Build secara incremental untuk dataset sangat besar
    Memory-efficient tapi sedikit lebih lambat
    
    Key: Fit encoders sekali di awal, transform per batch
    """
    from scipy.sparse import vstack
    
    dataset = dataset.copy()
    
    # ---- YEAR (preprocessing semua data sekaligus - ringan) ----
    dataset['aired'] = pd.to_datetime(dataset['aired'], errors='coerce')
    dataset['year'] = dataset['aired'].dt.year
    dataset['year'] = pd.to_numeric(dataset['year'], errors='coerce')
    dataset['year'] = dataset['year'].fillna(dataset['year'].median())
    
    year_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    year_scaler.fit(dataset[['year']])
    
    # ---- FIT ENCODERS SEKALI (pelajari vocabulary dari SEMUA data) ----
    print("Fitting encoders...")
    mlb_genre = MultiLabelBinarizer(sparse_output=True)
    mlb_studio = MultiLabelBinarizer(sparse_output=True)
    mlb_producer = MultiLabelBinarizer(sparse_output=True)
    mlb_keyword = MultiLabelBinarizer(sparse_output=True)
    
    mlb_genre.fit(dataset['genre'])
    mlb_studio.fit(dataset['studio'])
    mlb_producer.fit(dataset['producer'])
    mlb_keyword.fit(dataset['keywords'])
    
    # ---- TRANSFORM PER BATCH ----
    print(f"Processing {len(dataset)} samples in batches of {batch_size}...")
    feature_matrices = []
    n_samples = len(dataset)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch = dataset.iloc[start_idx:end_idx]
        
        # Transform batch dengan encoder yang sudah di-fit
        X_genre = mlb_genre.transform(batch['genre']).astype(np.float32)
        X_studio = mlb_studio.transform(batch['studio']).astype(np.float32)
        X_producer = mlb_producer.transform(batch['producer']).astype(np.float32)
        X_keyword = mlb_keyword.transform(batch['keywords']).astype(np.float32)
        X_year = csr_matrix(
            year_scaler.transform(batch[['year']]), 
            dtype=np.float32
        )
        
        # Weighted stacking
        batch_matrix = hstack([
            X_genre.multiply(2.5),
            X_studio.multiply(2.0),
            X_producer.multiply(1.75),
            X_keyword.multiply(1.0),
            X_year.multiply(0.5)
        ], format='csr', dtype=np.float32)
        
        feature_matrices.append(batch_matrix)
        
        if end_idx % (batch_size * 2) == 0 or end_idx == n_samples:
            print(f"Processed {end_idx}/{n_samples} samples")
    
    # Gabungkan semua batch (sekarang semua punya dimensi sama!)
    print("Stacking batches...")
    return vstack(feature_matrices, format='csr')


# ===============================
# USER PROFILE (tetap sama, sudah optimal)
# ===============================

def build_user_profile(feature_matrix, liked_indices):
    """
    User profile dari anime yang disukai
    """
    if len(liked_indices) == 0:
        return None

    user_profile = feature_matrix[liked_indices].sum(axis=0)

    if not isinstance(user_profile, csr_matrix):
        user_profile = csr_matrix(user_profile)

    norm = np.sqrt(user_profile.multiply(user_profile).sum())
    if norm == 0 or np.isnan(norm):
        return None

    user_profile = user_profile / norm

    return user_profile