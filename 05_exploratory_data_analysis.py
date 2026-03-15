"""
05_exploratory_data_analysis.py
================================
DATA 210P Final Project - Exploratory Data Analysis
Author: Rohan Ganesh
Date: March 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plasma = plt.cm.plasma

os.makedirs('outputs', exist_ok=True)

FIGURE_DPI = 150
FIGURE_FORMAT = 'png'

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS - ANIME SUCCESS PREDICTION")
print("=" * 60)

print("\n[1/8] Loading data...")
df = pd.read_csv('data/anime-dataset-2023.csv')
print(f"     Loaded {len(df):,} anime entries")

df['Score_clean'] = pd.to_numeric(df['Score'], errors='coerce')
df_scored = df[df['Score_clean'].notna() & (df['Score_clean'] > 0)].copy()
print(f"     {len(df_scored):,} anime with valid scores")

df_scored['Type_clean'] = df_scored['Type'].fillna('Unknown')
df_scored['Source_clean'] = df_scored['Source'].fillna('Unknown')

def extract_year(aired):
    """Extract the year from the Aired string."""
    if pd.isna(aired):
        return np.nan
    try:
        parts = str(aired).split()
        for part in parts:
            if part.isdigit() and len(part) == 4:
                return int(part)
    except:
        pass
    return np.nan

df_scored['Year'] = df_scored['Aired'].apply(extract_year)

df_scored['Episodes_clean'] = pd.to_numeric(df_scored['Episodes'], errors='coerce')

df_scored['is_highly_rated'] = (df_scored['Score_clean'] >= 8.0).astype(int)

def parse_genres(genres_str):
    """Parse comma-separated genres into a list."""
    if pd.isna(genres_str):
        return []
    return [g.strip() for g in str(genres_str).split(',')]

df_scored['genre_list'] = df_scored['Genres'].apply(parse_genres)

print(f"     Score range: {df_scored['Score_clean'].min():.2f} - {df_scored['Score_clean'].max():.2f}")
print(f"     Mean score: {df_scored['Score_clean'].mean():.2f} (SD = {df_scored['Score_clean'].std():.2f})")
print(f"     Highly rated (≥8.0): {df_scored['is_highly_rated'].sum():,} ({df_scored['is_highly_rated'].mean()*100:.1f}%)")

print("\n[2/8] Creating Figure 1: Score Distribution...")

fig, ax = plt.subplots(figsize=(10, 6))

n, bins, patches = ax.hist(df_scored['Score_clean'], bins=40, edgecolor='white', alpha=0.9)

bin_centers = 0.5 * (bins[:-1] + bins[1:])
col = (bin_centers - bin_centers.min()) / (bin_centers.max() - bin_centers.min())
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', plasma(c))

ax.axvline(x=8.0, color='#00FF00', linestyle='--', linewidth=2.5, 
           label='Highly Rated Threshold (8.0)')
ax.axvline(x=df_scored['Score_clean'].mean(), color='white', linestyle='-', 
           linewidth=2.5, label=f'Mean ({df_scored["Score_clean"].mean():.2f})')

ax.set_xlabel('Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Anime Scores', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, facecolor='white', framealpha=0.9)
ax.set_xlim(1, 10)
ax.set_facecolor('#f5f5f5')

plt.tight_layout()
plt.savefig('outputs/fig1_score_distribution.png', dpi=FIGURE_DPI, 
            bbox_inches='tight', facecolor='white')
plt.close()
print("     Saved: outputs/fig1_score_distribution.png")

print("\n[3/8] Creating Figure 2: Score by Type...")

main_types = ['TV', 'Movie', 'OVA', 'ONA', 'Special', 'Music']
type_order = (df_scored[df_scored['Type_clean'].isin(main_types)]
              .groupby('Type_clean')['Score_clean']
              .median()
              .sort_values(ascending=False)
              .index.tolist())

fig, ax = plt.subplots(figsize=(10, 6))

plasma_colors = [plasma(i/len(type_order)) for i in range(len(type_order))]
df_plot = df_scored[df_scored['Type_clean'].isin(main_types)]

sns.boxplot(data=df_plot, x='Type_clean', y='Score_clean', 
            order=type_order, palette=plasma_colors, ax=ax)

ax.axhline(y=8.0, color='#00FF00', linestyle='--', linewidth=2, 
           alpha=0.8, label='Highly Rated (8.0)')

ax.set_xlabel('Anime Type', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Score Distribution by Anime Type', fontsize=14, fontweight='bold')
ax.legend(facecolor='white', framealpha=0.9)
ax.set_facecolor('#f5f5f5')

plt.tight_layout()
plt.savefig('outputs/fig2_score_by_type.png', dpi=FIGURE_DPI, 
            bbox_inches='tight', facecolor='white')
plt.close()
print("     Saved: outputs/fig2_score_by_type.png")

print("\n[4/8] Creating Figure 3: Score by Source...")

source_counts = df_scored['Source_clean'].value_counts()
top_sources = source_counts[source_counts > 100].index.tolist()[:10]

source_order = (df_scored[df_scored['Source_clean'].isin(top_sources)]
                .groupby('Source_clean')['Score_clean']
                .median()
                .sort_values(ascending=False)
                .index.tolist())

fig, ax = plt.subplots(figsize=(12, 6))

plasma_colors = [plasma(i/len(source_order)) for i in range(len(source_order))]
df_plot = df_scored[df_scored['Source_clean'].isin(top_sources)]

sns.boxplot(data=df_plot, x='Source_clean', y='Score_clean', 
            order=source_order, palette=plasma_colors, ax=ax)

ax.axhline(y=8.0, color='#00FF00', linestyle='--', linewidth=2, 
           alpha=0.8, label='Highly Rated (8.0)')

ax.set_xlabel('Source Material', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Score Distribution by Source Material', fontsize=14, fontweight='bold')
plt.xticks(rotation=30, ha='right')
ax.legend(facecolor='white', framealpha=0.9)
ax.set_facecolor('#f5f5f5')

plt.tight_layout()
plt.savefig('outputs/fig3_score_by_source.png', dpi=FIGURE_DPI, 
            bbox_inches='tight', facecolor='white')
plt.close()
print("     Saved: outputs/fig3_score_by_source.png")

print("\n[5/8] Creating Figure 4: Mean Score by Genre...")

all_genres = []
for idx, row in df_scored.iterrows():
    for genre in row['genre_list']:
        all_genres.append({'genre': genre, 'score': row['Score_clean']})

genre_df = pd.DataFrame(all_genres)

genre_means = genre_df.groupby('genre')['score'].agg(['mean', 'count']).reset_index()
genre_means = genre_means[genre_means['count'] > 200].sort_values('mean', ascending=True).tail(12)

fig, ax = plt.subplots(figsize=(10, 7))

colors = [plasma(i/len(genre_means)) for i in range(len(genre_means))]
bars = ax.barh(genre_means['genre'], genre_means['mean'], 
               color=colors, edgecolor='white', linewidth=0.5)

ax.axvline(x=df_scored['Score_clean'].mean(), color='white', linestyle='--', 
           linewidth=2, label=f'Overall Mean ({df_scored["Score_clean"].mean():.2f})')

ax.set_xlabel('Mean Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Genre', fontsize=12, fontweight='bold')
ax.set_title('Mean Score by Genre (Top 12 by Count)', fontsize=14, fontweight='bold')
ax.set_xlim(5.5, 7.5)
ax.legend(facecolor='gray', framealpha=0.7, labelcolor='white')
ax.set_facecolor('#f5f5f5')

for bar, val in zip(bars, genre_means['mean']):
    ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
            f'{val:.2f}', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/fig4_score_by_genre.png', dpi=FIGURE_DPI, 
            bbox_inches='tight', facecolor='white')
plt.close()
print("     Saved: outputs/fig4_score_by_genre.png")

print("\n[6/8] Creating Figure 5: Missing Data by Popularity...")

def check_has_score(x):
    """Check if a score value is valid."""
    if pd.isna(x):
        return False
    x_str = str(x).upper()
    if x_str == 'UNKNOWN' or x_str == '':
        return False
    try:
        return float(x) > 0
    except:
        return False

df['Members_clean'] = pd.to_numeric(df['Members'], errors='coerce')
df['has_score'] = df['Score'].apply(check_has_score)

df_with_members = df[df['Members_clean'].notna()].copy()
df_with_members['popularity_quartile'] = pd.qcut(
    df_with_members['Members_clean'], 
    q=4, 
    labels=['Q1\n(Least Popular)', 'Q2', 'Q3', 'Q4\n(Most Popular)']
)

missing_by_quartile = (df_with_members
                       .groupby('popularity_quartile')['has_score']
                       .apply(lambda x: (1 - x.mean()) * 100))

fig, ax = plt.subplots(figsize=(9, 6))

colors = [plasma(0.9), plasma(0.65), plasma(0.4), plasma(0.15)]
bars = ax.bar(missing_by_quartile.index, missing_by_quartile.values, 
              color=colors, edgecolor='white', linewidth=2)

ax.set_xlabel('Popularity Quartile', fontsize=12, fontweight='bold')
ax.set_ylabel('% Missing Scores', fontsize=12, fontweight='bold')
ax.set_title('Missing Scores by Popularity Quartile', fontsize=14, fontweight='bold')
ax.set_ylim(0, 110)
ax.set_facecolor('#f5f5f5')

for bar, val in zip(bars, missing_by_quartile.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
            f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/fig5_missing_by_popularity.png', dpi=FIGURE_DPI, 
            bbox_inches='tight', facecolor='white')
plt.close()
print("     Saved: outputs/fig5_missing_by_popularity.png")

print("\n[7/8] Creating Figure 6: Score vs Year...")

df_plot = df_scored.dropna(subset=['Year', 'Episodes_clean']).copy()
df_plot = df_plot[(df_plot['Year'] >= 1960) & (df_plot['Year'] <= 2024)]

year_means = df_plot.groupby('Year')['Score_clean'].agg(['mean', 'std', 'count']).reset_index()
year_means = year_means[year_means['count'] >= 10]  # Only years with 10+ anime

fig, ax = plt.subplots(figsize=(12, 6))

scatter = ax.scatter(df_plot['Year'], df_plot['Score_clean'], 
                     c=df_plot['Score_clean'], cmap='plasma', 
                     alpha=0.3, s=15, edgecolors='none')

ax.plot(year_means['Year'], year_means['mean'], color='white', linewidth=3)
ax.plot(year_means['Year'], year_means['mean'], color=plasma(0.7), 
        linewidth=2, label='Mean Score by Year')

z = np.polyfit(year_means['Year'], year_means['mean'], 1)
p = np.poly1d(z)
ax.plot(year_means['Year'], p(year_means['Year']), color=plasma(0.2), 
        linestyle='--', linewidth=2, label=f'Trend (slope: {z[0]:.3f}/year)')

ax.axhline(y=8.0, color='#00FF00', linestyle='--', linewidth=1.5, 
           alpha=0.7, label='Highly Rated (8.0)')

ax.set_xlabel('Release Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Anime Scores Over Time', fontsize=14, fontweight='bold')
ax.set_xlim(1985, 2025)
ax.set_ylim(1, 10)
ax.legend(loc='lower right', facecolor='white', framealpha=0.9)
ax.set_facecolor('#f5f5f5')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
cbar.set_label('Score', fontsize=10)

plt.tight_layout()
plt.savefig('outputs/fig6_score_vs_year.png', dpi=FIGURE_DPI, 
            bbox_inches='tight', facecolor='white')
plt.close()
print("     Saved: outputs/fig6_score_vs_year.png")

print("\n[8/8] Creating Figure 7: Score vs Episodes...")

df_eps = df_plot[df_plot['Episodes_clean'] <= 500].copy()

fig, ax = plt.subplots(figsize=(10, 6))

scatter = ax.scatter(df_eps['Episodes_clean'], df_eps['Score_clean'], 
                     c=df_eps['Score_clean'], cmap='plasma', 
                     alpha=0.4, s=20, edgecolors='none')

ax.axhline(y=8.0, color='#00FF00', linestyle='--', linewidth=1.5, 
           alpha=0.7, label='Highly Rated (8.0)')
ax.axhline(y=df_eps['Score_clean'].mean(), color='white', linestyle='-', 
           linewidth=2, label=f'Overall Mean ({df_eps["Score_clean"].mean():.2f})')

ax.set_xlabel('Number of Episodes', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Anime Scores vs Episode Count', fontsize=14, fontweight='bold')
ax.set_xlim(0, 200)
ax.set_ylim(1, 10)
ax.legend(loc='lower right', facecolor='white', framealpha=0.9)
ax.set_facecolor('#f5f5f5')

cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
cbar.set_label('Score', fontsize=10)

plt.tight_layout()
plt.savefig('outputs/fig7_score_vs_episodes.png', dpi=FIGURE_DPI, 
            bbox_inches='tight', facecolor='white')
plt.close()
print("     Saved: outputs/fig7_score_vs_episodes.png")

print("\n" + "=" * 60)
print("EDA SUMMARY")
print("=" * 60)

print(f"\nDataset Overview:")
print(f"  - Total anime in raw data: {len(df):,}")
print(f"  - Anime with valid scores: {len(df_scored):,}")
print(f"  - Missing scores: {len(df) - len(df_scored):,} ({(len(df) - len(df_scored))/len(df)*100:.1f}%)")

print(f"\nScore Statistics:")
print(f"  - Mean: {df_scored['Score_clean'].mean():.2f}")
print(f"  - Std Dev: {df_scored['Score_clean'].std():.2f}")
print(f"  - Min: {df_scored['Score_clean'].min():.2f}")
print(f"  - Max: {df_scored['Score_clean'].max():.2f}")
print(f"  - Highly rated (≥8.0): {df_scored['is_highly_rated'].sum():,} ({df_scored['is_highly_rated'].mean()*100:.1f}%)")

print(f"\nType Distribution (top 5):")
for t, count in df_scored['Type_clean'].value_counts().head(5).items():
    print(f"  - {t}: {count:,}")

print(f"\nSource Distribution (top 5):")
for s, count in df_scored['Source_clean'].value_counts().head(5).items():
    print(f"  - {s}: {count:,}")

print("\n" + "=" * 60)
print("All 7 figures saved to outputs/ folder")
print("=" * 60)
