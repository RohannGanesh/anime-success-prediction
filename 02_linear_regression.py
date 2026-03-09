"""
DATA 210P Final Project: Predicting Anime Success
Step 2: Linear Regression Modeling

Author: Rohan
Date: Winter 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages
import warnings
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')


def get_genre_columns(df):
    
    return [col for col in df.columns if col.startswith('genre_')]


def build_incremental_models(df):
    
    
    print("="*60)
    print("BUILDING LINEAR REGRESSION MODELS")
    print("="*60)
    
    genre_cols = get_genre_columns(df)
    genre_formula = ' + '.join(genre_cols)
    
    models = {}
    
    print("\n[M0] Null Model (Intercept Only)...")
    models['M0_Null'] = smf.ols('Score_clean ~ 1', data=df).fit()
    
    print("[M1] Adding Type...")
    models['M1_Type'] = smf.ols('Score_clean ~ C(Type_clean)', data=df).fit()
    
    print("[M2] Adding Source...")
    models['M2_Source'] = smf.ols(
        'Score_clean ~ C(Type_clean) + C(Source_clean)', 
        data=df
    ).fit()
    
    print("[M3] Adding Episodes + Year...")
    models['M3_EpsYear'] = smf.ols(
        'Score_clean ~ C(Type_clean) + C(Source_clean) + Episodes_log + Year', 
        data=df
    ).fit()
    
    print("[M4] Adding Genres...")
    formula4 = f'Score_clean ~ C(Type_clean) + C(Source_clean) + Episodes_log + Year + {genre_formula}'
    models['M4_Genres'] = smf.ols(formula4, data=df).fit()
    
    print("[M5] Adding Studios (Full Model)...")
    formula5 = f'Score_clean ~ C(Type_clean) + C(Source_clean) + Episodes_log + Year + {genre_formula} + C(Studio_clean)'
    models['M5_Full'] = smf.ols(formula5, data=df).fit()
    
    return models


def compare_models(models):
    
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    comparison = []
    for name, model in models.items():
        comparison.append({
            'Model': name,
            'R²': model.rsquared,
            'Adj R²': model.rsquared_adj,
            'AIC': model.aic,
            'BIC': model.bic,
            'Parameters': int(model.df_model) + 1
        })
    
    df_comp = pd.DataFrame(comparison)
    print("\n", df_comp.to_string(index=False))
    
    return df_comp


def run_diagnostics(model, df):
   
    
    print("\n" + "="*60)
    print("MODEL DIAGNOSTICS")
    print("="*60)
    
    residuals = model.resid
    fitted = model.fittedvalues
    
    print("\n1. NORMALITY OF RESIDUALS")
    subsample = residuals.sample(min(5000, len(residuals)), random_state=42)
    shapiro_stat, shapiro_p = stats.shapiro(subsample)
    print(f"   Shapiro-Wilk: W = {shapiro_stat:.4f}, p = {shapiro_p:.4e}")
    print(f"   Result: {'⚠️ Violated' if shapiro_p < 0.05 else '✓ Satisfied'}")
    
    print("\n2. HOMOSCEDASTICITY")
    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, model.model.exog)
    print(f"   Breusch-Pagan: LM = {bp_stat:.2f}, p = {bp_p:.4e}")
    print(f"   Result: {'⚠️ Heteroscedasticity detected' if bp_p < 0.05 else '✓ Satisfied'}")
    
    print("\n3. INDEPENDENCE")
    dw = durbin_watson(residuals)
    print(f"   Durbin-Watson: {dw:.4f}")
    print(f"   Result: {'✓ No autocorrelation' if 1.5 < dw < 2.5 else '⚠️ Potential autocorrelation'}")
    
    return {
        'shapiro_p': shapiro_p,
        'bp_p': bp_p,
        'durbin_watson': dw
    }


def plot_diagnostics(model, df, output_path='outputs/linear_regression_diagnostics.pdf'):
  
    
    print(f"\nCreating diagnostic plots: {output_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fitted = model.fittedvalues
    residuals = model.resid
    std_resid = model.get_influence().resid_studentized_internal
    
    pdf = PdfPages(output_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Linear Regression Diagnostics', fontsize=16, fontweight='bold')
    
    axes[0, 0].scatter(fitted, residuals, alpha=0.3, s=10)
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    
    stats.probplot(std_resid, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot')
    
    axes[1, 0].scatter(fitted, np.sqrt(np.abs(std_resid)), alpha=0.3, s=10)
    axes[1, 0].set_xlabel('Fitted Values')
    axes[1, 0].set_ylabel('√|Standardized Residuals|')
    axes[1, 0].set_title('Scale-Location')
    
    axes[1, 1].hist(std_resid, bins=50, density=True, alpha=0.7, edgecolor='white')
    x = np.linspace(-4, 4, 100)
    axes[1, 1].plot(x, stats.norm.pdf(x), 'r-', linewidth=2)
    axes[1, 1].set_xlabel('Standardized Residuals')
    axes[1, 1].set_title('Residual Distribution')
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    ax.text(0.5, 0.5, f'R² = {model.rsquared:.4f}\nAdj R² = {model.rsquared_adj:.4f}',
            ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_title('Model Fit')
    ax.axis('off')
    
    ax = axes[0, 1]
    ax.text(0.5, 0.5, f'AIC = {model.aic:.2f}\nBIC = {model.bic:.2f}',
            ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_title('Information Criteria')
    ax.axis('off')
    
    coefs = model.params.drop('Intercept').sort_values()
    
    ax = axes[1, 0]
    top_neg = coefs.head(8)
    ax.barh(range(len(top_neg)), top_neg.values, color='#ef4444')
    ax.set_yticks(range(len(top_neg)))
    ax.set_yticklabels([x[:25] for x in top_neg.index], fontsize=8)
    ax.set_title('Top Negative Effects')
    
    ax = axes[1, 1]
    top_pos = coefs.tail(8)
    ax.barh(range(len(top_pos)), top_pos.values, color='#10b981')
    ax.set_yticks(range(len(top_pos)))
    ax.set_yticklabels([x[:25] for x in top_pos.index], fontsize=8)
    ax.set_title('Top Positive Effects')
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()
    
    pdf.close()
    print(f"✓ Saved: {output_path}")


def print_key_findings(model):
    
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    sig = model.pvalues < 0.05
    coefs = model.params[sig].sort_values()
    
    print(f"\nModel R² = {model.rsquared:.4f}")
    print(f"The model explains {model.rsquared*100:.1f}% of variance in anime scores.")
    
    print("\nTop NEGATIVE effects (lower scores):")
    for name, val in coefs.head(5).items():
        if name != 'Intercept':
            print(f"  {name}: {val:+.4f}")
    
    print("\nTop POSITIVE effects (higher scores):")
    for name, val in coefs.tail(5).items():
        if name != 'Intercept':
            print(f"  {name}: {val:+.4f}")


def main():
  
    DATA_PATH = "data/anime_modeling_data.csv"
    OUTPUT_PATH = "outputs/linear_regression_diagnostics.pdf"
    
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Prepared data not found at {DATA_PATH}")
        print("Please run 01_data_preparation.py first")
        return
    
    print(f"Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Sample size: {len(df):,} anime")
    
    models = build_incremental_models(df)
    
    comparison = compare_models(models)
    
    best_model = models['M5_Full']
    
    diagnostics = run_diagnostics(best_model, df)
    
    plot_diagnostics(best_model, df, OUTPUT_PATH)
    
    print_key_findings(best_model)
    
    with open('outputs/linear_regression_summary.txt', 'w') as f:
        f.write(best_model.summary().as_text())
    print("\n✓ Model summary saved to: outputs/linear_regression_summary.txt")
    
    return models, comparison


if __name__ == "__main__":
    models, comparison = main()
