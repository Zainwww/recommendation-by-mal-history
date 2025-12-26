import pandas as pd
import ast
import gc

DATASET_PATH = './app/static/data/dataset.csv'

_dataset = None

# Parsing genre & studio
def parse_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return [i.strip() for i in x.split(',')]
    return []

def get_dataset():
    global _dataset
    if _dataset is None:
        df = pd.read_csv(DATASET_PATH)
        df['mal_id'] = pd.to_numeric(df['mal_id'], errors='coerce')
        df = df.rename(columns={'score': 'mal_score'})
        
        df['studio'] = df['studio'].apply(parse_list)
        df['genre'] = df['genre'].apply(parse_list)
        df['producer'] = df['producer'].apply(parse_list)
        df['keywords'] = df['keywords'].apply(parse_list)
        df['theme'] = df['theme'].apply(parse_list)

        if df['demographic'].dtype == 'object':
            df['demographic'] = df['demographic'].apply(parse_list)

        _dataset = df

    return _dataset.copy(deep=False)

def clear_dataset_cache():
    global _dataset
    _dataset = None
    gc.collect()
