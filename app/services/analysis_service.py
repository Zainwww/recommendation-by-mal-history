import pandas as pd
import numpy as np
import pandas as pd

def analysis_anime(data):
    df = pd.DataFrame(data)
    df.rename(columns={'id': 'mal_id'}, inplace=True)
    df.rename(columns={'score': 'my_score'}, inplace=True)

    file_path = './app/static/data/dataset.csv'
    dataset = pd.read_csv(file_path)

    # set type mal_id
    df['mal_id'] = pd.to_numeric(df['mal_id'], errors='coerce')
    dataset['mal_id'] = pd.to_numeric(dataset['mal_id'], errors='coerce')

    # drop duplicate colomns
    cols_to_drop = [col for col in df.columns if col in dataset.columns and col != 'mal_id']
    df_clean = dataset.drop(columns=cols_to_drop+['image_url'])

    merged_df = pd.merge(df, df_clean, on='mal_id', how='inner')
    merged_df = merged_df.drop(['Unnamed: 0'], axis=1)

    genre = fetch_genre(merged_df)
    studio = fetch_studio(merged_df)
    producer = fetch_producer(merged_df)
    demographic = fetch_demographic(merged_df)
    theme = fetch_theme(merged_df)
    anime_time = fetch_anime_time(merged_df)
    analysis = fetch_analysis(merged_df)

    return {
        "genre": genre,
        "studio": studio,
        "producer": producer,
        "demographic": demographic,
        "theme": theme,
        "anime_time": anime_time,
        "recommendation": analysis
    }

def fetch_studio(df):
    df['studio'] = df['studio'].str.split(',')
    df['studio'] = df['studio'].apply(lambda x: eval(x) if isinstance(x, str) else x)

    studio_df = df.explode('studio')['studio'].value_counts().reset_index()
    studio_df.columns = ['studios', 'count']

    return studio_df.to_dict(orient='records')

def fetch_genre(df):
    df['genre'] = df['genre'].str.split(',')
    df['genre'] = df['genre'].apply(lambda x: eval(x) if isinstance(x, str) else x)

    genre_df = df.explode('genre')['genre'].value_counts().reset_index()
    genre_df.columns = ['genres', 'count']

    return genre_df.to_dict(orient='records')

def fetch_producer(df):
    df['producer'] = df['producer'].str.split(',')
    df['producer'] = df['producer'].apply(lambda x: eval(x) if isinstance(x, str) else x)

    producer_df = df.explode('producer')['producer'].value_counts().reset_index()
    producer_df.columns = ['producers', 'count']

    return producer_df.to_dict(orient='records')

def fetch_demographic(df):
    df['demographic'] = df['demographic'].str.split(',')
    df['demographic'] = df['demographic'].apply(lambda x: eval(x) if isinstance(x, str) else x)

    demographic_df = df.explode('demographic')['demographic'].value_counts().reset_index()
    demographic_df.columns = ['demographics', 'count']
    demographic_df = demographic_df[demographic_df['demographics'] != '-']

    return demographic_df.to_dict(orient='records')

def fetch_theme(df):
    df['theme'] = df['theme'].str.split(',')
    df['theme'] = df['theme'].apply(lambda x: eval(x) if isinstance(x, str) else x)

    theme_df = df.explode('theme')['theme'].value_counts().reset_index()
    theme_df.columns = ['themes', 'count']
    theme_df = theme_df[theme_df['themes'] != '-']

    return theme_df.to_dict(orient='records')
def fetch_anime_time(df):
    df['premiered'] = (df['premiered'].str.replace('  ', ' ', regex=False).str.strip())
    df = df[~df['premiered'].isin(['-', '?'])]
    df['season'] = df['premiered'].str.extract(r'^(Fall|Spring|Summer|Winter)')
    df['year'] = df['premiered'].str.extract(r'(\d{4})')
    df = df.dropna(subset=['season', 'year'])
    count_df = (df.groupby(['year', 'season']).size().reset_index(name='count'))
    season_order = ['Fall', 'Spring', 'Summer', 'Winter']
    full_index = pd.MultiIndex.from_product([count_df['year'].unique(), season_order],names=['year', 'season'])

    final_df = (count_df.set_index(['year', 'season']).reindex(full_index, fill_value=0).reset_index())
    final_df['time'] = final_df['season'] + ' ' + final_df['year']
    final_df['season'] = pd.Categorical(final_df['season'],categories=season_order,ordered=True)

    final_df = final_df.sort_values(['year', 'season'])
    time_df = final_df[['time', 'count']].reset_index()
    time_df = time_df.drop(columns=['index'])

    return time_df.to_dict(orient='records')

def fetch_analysis(df):
    return []

def main():
    while True:
        analysis_anime()

if __name__ == "__main__":
    main()