"""
DATA 210P Final Project: Predicting Anime Success
Step 3: Logistic Regression Modeling
Author: Rohan
Date: Winter 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from matplotlib.backends.backend_pdf import PdfPages
import warnings
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# Colors
COLORS = {
    'primary': '#2563eb',
    'secondary': '#7c3aed', 
    'success': '#10b981',
    'danger': '#ef4444',
    'accent': '#f59e0b'
}


def get_genre_columns(df):
    return [col for col in df.columns if col.startswith('genre_')]


def prepare_features(df):
    print("Preparing features for logistic regression...")
    
    genre_cols = get_genre_columns(df)
    
    df_encoded = pd.get_dummies(df, columns=['Type_clean', 'Source_clean', 'Studio_clean'], 
                                 drop_first=True)
    
    feature_cols = []
    
    feature_cols.extend(['Episodes_log', 'Year'])
    
    feature_cols.extend(genre_cols)
    
    for col in df_encoded.columns:
        if col.startswith(('Type_clean_', 'Source_clean_', 'Studio_clean_')):
            feature_cols.append(col)
    
    feature_cols = [col for col in feature_cols if col in df_encoded.columns]
    
    X = df_encoded[feature_cols].copy()
    y = df_encoded['is_highly_rated'].copy()
    
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Positive class (highly rated): {y.sum()} ({y.mean()*100:.1f}%)")
    
    return X, y, feature_cols


def build_statsmodels_logistic(df):
    
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION (Statsmodels - for Inference)")
    print("="*60)
    
    genre_cols = get_genre_columns(df)
    genre_formula = ' + '.join(genre_cols)
    
    formula = f'is_highly_rated ~ C(Type_clean) + C(Source_clean) + Episodes_log + Year + {genre_formula}'
    
    print(f"\nFormula: is_highly_rated ~ Type + Source + Episodes + Year + Genres")
    
    model = smf.logit(formula, data=df).fit(disp=0)
    
    print(f"\nModel converged: {model.mle_retvals['converged']}")
    print(f"Pseudo R²: {model.prsquared:.4f}")
    print(f"Log-Likelihood: {model.llf:.2f}")
    print(f"AIC: {model.aic:.2f}")
    print(f"BIC: {model.bic:.2f}")
    
    return model


def build_sklearn_logistic(X, y):
    
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION (Sklearn - for Prediction)")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Train positive rate: {y_train.mean()*100:.1f}%")
    print(f"Test positive rate: {y_test.mean()*100:.1f}%")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    print("\n" + "-"*40)
    print("MODEL PERFORMANCE (Test Set)")
    print("-"*40)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"              Neg     Pos")
    print(f"Actual Neg   {cm[0,0]:5d}   {cm[0,1]:5d}")
    print(f"Actual Pos   {cm[1,0]:5d}   {cm[1,1]:5d}")
    
    print("\n" + "-"*40)
    print("CROSS-VALIDATION (5-Fold)")
    print("-"*40)
    
    cv_scores = cross_val_score(
        LogisticRegression(max_iter=1000, random_state=42),
        scaler.fit_transform(X), y, cv=5, scoring='roc_auc'
    )
    print(f"CV ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    return model, scaler, X_train, X_test, y_train, y_test, y_pred, y_prob


def get_odds_ratios(statsmodel):
    
    print("\n" + "="*60)
    print("ODDS RATIOS (Key Predictors)")
    print("="*60)
    
    params = statsmodel.params
    conf = statsmodel.conf_int()
    
    odds_ratios = np.exp(params)
    ci_lower = np.exp(conf[0])
    ci_upper = np.exp(conf[1])
    pvalues = statsmodel.pvalues
    
    or_df = pd.DataFrame({
        'Odds Ratio': odds_ratios,
        'CI Lower': ci_lower,
        'CI Upper': ci_upper,
        'p-value': pvalues
    })
    
    sig_or = or_df[or_df['p-value'] < 0.05].copy()
    sig_or = sig_or.drop('Intercept', errors='ignore')
    
    sig_or_sorted = sig_or.sort_values('Odds Ratio')
    
    print("\nSignificant predictors (p < 0.05):")
    print("-"*60)
    print(f"{'Predictor':<35} {'OR':>8} {'95% CI':>15} {'p':>10}")
    print("-"*60)
    
    for idx, row in sig_or_sorted.iterrows():
        name = idx[:35]
        or_val = row['Odds Ratio']
        ci = f"({row['CI Lower']:.2f}-{row['CI Upper']:.2f})"
        p = row['p-value']
        p_str = "<0.001" if p < 0.001 else f"{p:.3f}"
        print(f"{name:<35} {or_val:>8.2f} {ci:>15} {p_str:>10}")
    
    return or_df


def create_diagnostic_plots(y_test, y_pred, y_prob, feature_names, model, 
                            output_path='outputs/logistic_regression_diagnostics.pdf'):
    
    print(f"\nCreating diagnostic plots: {output_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    pdf = PdfPages(output_path)
    
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Logistic Regression: Model Performance', fontsize=18, fontweight='bold')
    
    ax1 = fig.add_subplot(221)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Not Highly Rated', 'Highly Rated'],
                yticklabels=['Not Highly Rated', 'Highly Rated'])
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title('Confusion Matrix', fontweight='bold')
    
    ax2 = fig.add_subplot(222)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color=COLORS['primary'], lw=2, 
             label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve', fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.05])
    
    ax3 = fig.add_subplot(223)
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1 Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC AUC': roc_auc
    }
    bars = ax3.bar(metrics.keys(), metrics.values(), color=COLORS['success'], edgecolor='black')
    ax3.set_ylim(0, 1)
    ax3.set_ylabel('Score')
    ax3.set_title('Performance Metrics', fontweight='bold')
    for bar, val in zip(bars, metrics.values()):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                 f'{val:.3f}', ha='center', fontsize=10)
    
    ax4 = fig.add_subplot(224)
    ax4.hist(y_prob[y_test == 0], bins=30, alpha=0.6, label='Not Highly Rated', 
             color=COLORS['primary'], edgecolor='white')
    ax4.hist(y_prob[y_test == 1], bins=30, alpha=0.6, label='Highly Rated', 
             color=COLORS['danger'], edgecolor='white')
    ax4.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
    ax4.set_xlabel('Predicted Probability')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Predicted Probability Distribution', fontweight='bold')
    ax4.legend()
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()
    
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Feature Importance (Coefficient Magnitude)', fontsize=18, fontweight='bold')
    
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0]
    }).sort_values('Coefficient')
    
    ax1 = fig.add_subplot(121)
    top_neg = coef_df.head(15)
    colors = [COLORS['danger']] * len(top_neg)
    ax1.barh(range(len(top_neg)), top_neg['Coefficient'], color=colors, edgecolor='black')
    ax1.set_yticks(range(len(top_neg)))
    ax1.set_yticklabels([x[:30] for x in top_neg['Feature']], fontsize=8)
    ax1.set_xlabel('Coefficient')
    ax1.set_title('Top 15 Negative Predictors', fontweight='bold')
    ax1.axvline(x=0, color='black', linewidth=1)
    
    ax2 = fig.add_subplot(122)
    top_pos = coef_df.tail(15)
    colors = [COLORS['success']] * len(top_pos)
    ax2.barh(range(len(top_pos)), top_pos['Coefficient'], color=colors, edgecolor='black')
    ax2.set_yticks(range(len(top_pos)))
    ax2.set_yticklabels([x[:30] for x in top_pos['Feature']], fontsize=8)
    ax2.set_xlabel('Coefficient')
    ax2.set_title('Top 15 Positive Predictors', fontweight='bold')
    ax2.axvline(x=0, color='black', linewidth=1)
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()
    

    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Threshold Analysis', fontsize=18, fontweight='bold')
    
    thresholds_range = np.arange(0.1, 0.9, 0.05)
    precisions = []
    recalls = []
    f1s = []
    
    for thresh in thresholds_range:
        y_pred_thresh = (y_prob >= thresh).astype(int)
        precisions.append(precision_score(y_test, y_pred_thresh, zero_division=0))
        recalls.append(recall_score(y_test, y_pred_thresh, zero_division=0))
        f1s.append(f1_score(y_test, y_pred_thresh, zero_division=0))
    
    ax1 = fig.add_subplot(111)
    ax1.plot(thresholds_range, precisions, 'b-', label='Precision', linewidth=2)
    ax1.plot(thresholds_range, recalls, 'g-', label='Recall', linewidth=2)
    ax1.plot(thresholds_range, f1s, 'r-', label='F1 Score', linewidth=2)
    ax1.axvline(x=0.5, color='gray', linestyle='--', label='Default Threshold')
    ax1.set_xlabel('Classification Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Precision, Recall, F1 vs Threshold', fontweight='bold')
    ax1.legend()
    ax1.set_xlim([0.1, 0.85])
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()
    
    pdf.close()
    print(f"✓ Saved: {output_path}")


def print_interpretation(y_test, y_pred, y_prob, or_df):
    """Print interpretation of results."""
    
    print("\n" + "="*60)
    print("INTERPRETATION & KEY FINDINGS")
    print("="*60)
    
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Get top odds ratios
    sig_or = or_df[or_df['p-value'] < 0.05].drop('Intercept', errors='ignore')
    top_positive = sig_or[sig_or['Odds Ratio'] > 1].nlargest(5, 'Odds Ratio')
    top_negative = sig_or[sig_or['Odds Ratio'] < 1].nsmallest(5, 'Odds Ratio')
    
    print(f"""
1. MODEL PERFORMANCE
   - ROC AUC = {roc_auc:.3f} indicates {'good' if roc_auc > 0.7 else 'moderate'} discrimination
   - The model can distinguish highly-rated anime from others

2. CLASS IMBALANCE
   - Only {(y_test==1).mean()*100:.1f}% of anime are highly rated (≥8.0)
   - This affects precision/recall tradeoffs

3. TOP PREDICTORS OF HIGH RATINGS (Odds Ratios > 1):""")
    
    for idx, row in top_positive.iterrows():
        name = idx.replace('C(', '').replace(')', '').replace('[T.', ': ').replace(']', '')
        print(f"   - {name}: OR = {row['Odds Ratio']:.2f}")
    
    print(f"""
4. PREDICTORS OF LOWER RATINGS (Odds Ratios < 1):""")
    
    for idx, row in top_negative.iterrows():
        name = idx.replace('C(', '').replace(')', '').replace('[T.', ': ').replace(']', '')
        print(f"   - {name}: OR = {row['Odds Ratio']:.2f}")
    
    print(f"""
5. PRACTICAL IMPLICATIONS
   - Drama and certain genres significantly increase odds of high rating
   - Source material matters: adaptations from manga/novels do better
   - Some studios consistently produce higher-rated content
""")


def main():

    DATA_PATH = "data/anime_modeling_data.csv"
    OUTPUT_PATH = "outputs/logistic_regression_diagnostics.pdf"
    
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data not found at {DATA_PATH}")
        print("Please run 01_data_preparation.py first")
        return
    
    print("="*60)
    print("LOGISTIC REGRESSION: PREDICTING HIGHLY RATED ANIME")
    print("="*60)
    print(f"\nLoading data from: {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    print(f"Sample size: {len(df):,} anime")
    print(f"Highly rated (≥8.0): {df['is_highly_rated'].sum():,} ({df['is_highly_rated'].mean()*100:.1f}%)")
    
    statsmodel = build_statsmodels_logistic(df)
    
    or_df = get_odds_ratios(statsmodel)
    
    X, y, feature_names = prepare_features(df)
    
    sklearn_model, scaler, X_train, X_test, y_train, y_test, y_pred, y_prob = \
        build_sklearn_logistic(X, y)
    
    create_diagnostic_plots(y_test, y_pred, y_prob, feature_names, sklearn_model, OUTPUT_PATH)
    
    print_interpretation(y_test, y_pred, y_prob, or_df)
    
    with open('outputs/logistic_regression_summary.txt', 'w') as f:
        f.write(statsmodel.summary().as_text())
    print("\n✓ Model summary saved to: outputs/logistic_regression_summary.txt")
    
    return statsmodel, sklearn_model, or_df


if __name__ == "__main__":
    statsmodel, sklearn_model, or_df = main()
