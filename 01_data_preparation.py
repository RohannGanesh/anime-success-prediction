"""
DATA 210P Final Project: Predicting Anime Success
Step 1: Data Preparation
Author: Rohan
Date: Winter 2026
"""

import pandas as pd
import numpy as np
import re
import os

def load_and_prepare_data(filepath):
    print("="*60)
    print("STEP 1: DATA PREPARATION")
    print("="*60)
  
    print(f"\nLoading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Raw dataset: {len(df):,} rows, {len(df.columns)} columns")
    

    print("\n[1/7] Creating outcome variables...")
    
    df['Score_clean'] = pd.to_numeric(df['Score'], errors='coerce')
    
    df_model = df[df['Score_clean'].notna()].copy()
    print(f"  Anime with valid scores: {len(df_model):,}")

    df_model['is_highly_rated'] = (df_model['Score_clean'] >= 8.0).astype(int)
    print(f"  Highly rated (≥8.0): {df_model['is_highly_rated'].sum():,} ({df_model['is_highly_rated'].mean()*100:.1f}%)")

    print("\n[2/7] Cleaning Type variable...")
    
    df_model['Type_clean'] = df_model['Type'].replace('UNKNOWN', 'Other')
    print(f"  Categories: {df_model['Type_clean'].unique().tolist()}")
    
    print("\n[3/7] Cleaning Source variable...")
    
    source_counts = df_model['Source'].value_counts()
    top_sources = source_counts[source_counts >= 100].index.tolist()
    df_model['Source_clean'] = df_model['Source'].apply(
        lambda x: x if x in top_sources else 'Other'
    )
    print(f"  Categories: {df_model['Source_clean'].nunique()}")
    
    print("\n[4/7] Creating genre dummy variables...")
    
    all_genres = []
    for genres in df_model['Genres']:
        if pd.notna(genres) and genres != 'UNKNOWN':
            for g in str(genres).split(','):
                g = g.strip()
                if g:
                    all_genres.append(g)
    
    genre_counts = pd.Series(all_genres).value_counts()
    top_genres = genre_counts.head(10).index.tolist()
    print(f"  Top 10 genres: {top_genres}")
    
    for genre in top_genres:
        col_name = f"genre_{genre.replace(' ', '_').replace('-', '_')}"
        df_model[col_name] = df_model['Genres'].str.contains(
            genre, case=False, na=False, regex=False
        ).astype(int)
    

    print("\n[5/7] Cleaning Studio variable...")
    
    def get_primary_studio(studio):
        if pd.isna(studio) or studio == 'UNKNOWN':
            return 'Unknown'
        return str(studio).split(',')[0].strip()
    
    df_model['Studio_primary'] = df_model['Studios'].apply(get_primary_studio)
    
    studio_counts = df_model['Studio_primary'].value_counts()
    top_studios = studio_counts[studio_counts >= 50].index.tolist()
    top_studios = [s for s in top_studios if s != 'Unknown'][:10]
    
    df_model['Studio_clean'] = df_model['Studio_primary'].apply(
        lambda x: x if x in top_studios else 'Other'
    )
    print(f"  Top studios: {top_studios}")
    

    print("\n[6/7] Cleaning Episodes variable...")
    
    df_model['Episodes_clean'] = pd.to_numeric(df_model['Episodes'], errors='coerce')
    df_model['Episodes_log'] = np.log1p(df_model['Episodes_clean'])
    print(f"  Valid episode counts: {df_model['Episodes_clean'].notna().sum():,}")

    print("\n[7/7] Extracting Year and Season...")
    
    def extract_year(aired):
        if pd.isna(aired) or aired == 'UNKNOWN':
            return np.nan
        years = re.findall(r'(19\d{2}|20\d{2})', str(aired))
        return int(years[0]) if years else np.nan
    
    def extract_season(premiered):
        if pd.isna(premiered) or premiered == 'UNKNOWN':
            return 'Unknown'
        premiered = str(premiered).lower()
        for season in ['winter', 'spring', 'summer', 'fall']:
            if season in premiered:
                return season.capitalize()
        return 'Unknown'
    
    df_model['Year'] = df_model['Aired'].apply(extract_year)
    df_model['Season'] = df_model['Premiered'].apply(extract_season)
    print(f"  Valid years: {df_model['Year'].notna().sum():,}")
    

    print("\n" + "="*60)
    print("FINAL DATASET")
    print("="*60)
    
    df_final = df_model.dropna(subset=['Score_clean', 'Episodes_clean', 'Year'])
    print(f"\nFinal sample size: {len(df_final):,} anime")
    print(f"Dropped: {len(df_model) - len(df_final):,} (missing Episodes or Year)")
    
    return df_final


def get_genre_columns(df):
  
    return [col for col in df.columns if col.startswith('genre_')]


def main():
    DATA_PATH = "data/anime-dataset-2023.csv"
    OUTPUT_PATH = "data/anime_modeling_data.csv"
    
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: File not found at {DATA_PATH}")
        print("Please update DATA_PATH to point to your anime-dataset-2023.csv file")
        return None
    
    df = load_and_prepare_data(DATA_PATH)
    
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Saved prepared data to: {OUTPUT_PATH}")
    
    return df


if __name__ == "__main__":
    df = main()
