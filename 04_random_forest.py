"""
DATA 210P Final Project: Predicting Anime Success
Step 4: Random Forest & Model Comparison
Author: Rohan
Date: Winter 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import warnings
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

COLORS = {
    'primary': '#2563eb',
    'secondary': '#7c3aed',
    'success': '#10b981',
    'danger': '#ef4444',
    'accent': '#f59e0b',
    'dark': '#1f2937'
}


def get_genre_columns(df):
    return [col for col in df.columns if col.startswith('genre_')]


def prepare_features(df):

    print("Preparing features...")
    
    genre_cols = get_genre_columns(df)
    
    df_encoded = pd.get_dummies(
        df, 
        columns=['Type_clean', 'Source_clean', 'Studio_clean'],
        drop_first=True
    )
    
    feature_cols = ['Episodes_log', 'Year']
    feature_cols.extend(genre_cols)
    
    for col in df_encoded.columns:
        if col.startswith(('Type_clean_', 'Source_clean_', 'Studio_clean_')):
            feature_cols.append(col)
    
    feature_cols = [col for col in feature_cols if col in df_encoded.columns]
    
    X = df_encoded[feature_cols].copy()
    y_continuous = df_encoded['Score_clean'].copy()
    y_binary = df_encoded['is_highly_rated'].copy()
    
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {X.shape[0]}")
    
    return X, y_continuous, y_binary, feature_cols


def random_forest_regression(X, y, feature_names):
    print("\n" + "="*60)
    print("RANDOM FOREST REGRESSION (Predicting Score)")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    print("\nTraining Random Forest (this may take a moment)...")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)
    
    print("\n" + "-"*40)
    print("PERFORMANCE METRICS")
    print("-"*40)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print(f"Train R²:  {train_r2:.4f}")
    print(f"Test R²:   {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE:  {test_mae:.4f}")
    
    print("\n" + "-"*40)
    print("CROSS-VALIDATION (5-Fold)")
    print("-"*40)
    
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
    print(f"CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n" + "-"*40)
    print("TOP 10 IMPORTANT FEATURES")
    print("-"*40)
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['Feature']:<30} {row['Importance']:.4f}")
    
    return rf, X_test, y_test, y_pred_test, importance_df


def random_forest_classification(X, y, feature_names):
    print("\n" + "="*60)
    print("RANDOM FOREST CLASSIFICATION (Predicting Highly Rated)")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    print(f"Positive rate - Train: {y_train.mean()*100:.1f}%, Test: {y_test.mean()*100:.1f}%")
    

    print("\nTraining Random Forest Classifier...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced', 
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    print("\n" + "-"*40)
    print("PERFORMANCE METRICS")
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
    
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='roc_auc')
    print(f"CV ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n" + "-"*40)
    print("TOP 10 IMPORTANT FEATURES")
    print("-"*40)
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['Feature']:<30} {row['Importance']:.4f}")
    
    return rf, X_test, y_test, y_pred, y_prob, importance_df


def compare_all_models(X, y_cont, y_bin, feature_names):
    """Compare Linear Regression, Logistic Regression, and Random Forest."""
    
    print("\n" + "="*60)
    print("MODEL COMPARISON: ALL MODELS")
    print("="*60)
    

    X_train, X_test, y_cont_train, y_cont_test = train_test_split(
        X, y_cont, test_size=0.2, random_state=42
    )
    _, _, y_bin_train, y_bin_test = train_test_split(
        X, y_bin, test_size=0.2, random_state=42, stratify=y_bin
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    print("\n" + "-"*40)
    print("REGRESSION: Predicting Score")
    print("-"*40)
    
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_cont_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_r2 = r2_score(y_cont_test, lr_pred)
    lr_rmse = np.sqrt(mean_squared_error(y_cont_test, lr_pred))
    
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train, y_cont_train)
    rf_reg_pred = rf_reg.predict(X_test)
    rf_reg_r2 = r2_score(y_cont_test, rf_reg_pred)
    rf_reg_rmse = np.sqrt(mean_squared_error(y_cont_test, rf_reg_pred))
    
    print(f"\n{'Model':<25} {'R²':>10} {'RMSE':>10}")
    print("-"*45)
    print(f"{'Linear Regression':<25} {lr_r2:>10.4f} {lr_rmse:>10.4f}")
    print(f"{'Random Forest':<25} {rf_reg_r2:>10.4f} {rf_reg_rmse:>10.4f}")
    
    results['regression'] = {
        'Linear Regression': {'R²': lr_r2, 'RMSE': lr_rmse},
        'Random Forest': {'R²': rf_reg_r2, 'RMSE': rf_reg_rmse}
    }
    
    print("\n" + "-"*40)
    print("CLASSIFICATION: Predicting Highly Rated")
    print("-"*40)
    
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_scaled, y_bin_train)
    log_pred = log_reg.predict(X_test_scaled)
    log_prob = log_reg.predict_proba(X_test_scaled)[:, 1]
    log_auc = roc_auc_score(y_bin_test, log_prob)
    log_f1 = f1_score(y_bin_test, log_pred, zero_division=0)
    
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=15, 
                                     class_weight='balanced', random_state=42, n_jobs=-1)
    rf_clf.fit(X_train, y_bin_train)
    rf_pred = rf_clf.predict(X_test)
    rf_prob = rf_clf.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_bin_test, rf_prob)
    rf_f1 = f1_score(y_bin_test, rf_pred, zero_division=0)
    
    print(f"\n{'Model':<25} {'ROC AUC':>10} {'F1 Score':>10}")
    print("-"*45)
    print(f"{'Logistic Regression':<25} {log_auc:>10.4f} {log_f1:>10.4f}")
    print(f"{'Random Forest':<25} {rf_auc:>10.4f} {rf_f1:>10.4f}")
    
    results['classification'] = {
        'Logistic Regression': {'ROC AUC': log_auc, 'F1': log_f1},
        'Random Forest': {'ROC AUC': rf_auc, 'F1': rf_f1}
    }
    
    results['models'] = {
        'lr': lr, 'rf_reg': rf_reg, 'log_reg': log_reg, 'rf_clf': rf_clf
    }
    results['data'] = {
        'X_test': X_test, 'X_test_scaled': X_test_scaled,
        'y_cont_test': y_cont_test, 'y_bin_test': y_bin_test,
        'log_prob': log_prob, 'rf_prob': rf_prob,
        'lr_pred': lr_pred, 'rf_reg_pred': rf_reg_pred
    }
    
    return results


def create_comparison_plots(results, rf_reg_importance, rf_clf_importance,
                            output_path='outputs/model_comparison.pdf'):
    """Create comprehensive model comparison plots."""
    
    print(f"\nCreating comparison plots: {output_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    pdf = PdfPages(output_path)
    
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Model Comparison: Regression (Predicting Score)', 
                 fontsize=18, fontweight='bold', y=0.96)
    
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    models = ['Linear\nRegression', 'Random\nForest']
    r2_scores = [
        results['regression']['Linear Regression']['R²'],
        results['regression']['Random Forest']['R²']
    ]
    colors = [COLORS['primary'], COLORS['success']]
    bars = ax1.bar(models, r2_scores, color=colors, edgecolor='black')
    ax1.set_ylabel('R²')
    ax1.set_title('R² Comparison', fontweight='bold')
    ax1.set_ylim(0, max(r2_scores) * 1.2)
    for bar, val in zip(bars, r2_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', fontsize=12, fontweight='bold')
    
    ax2 = fig.add_subplot(gs[0, 1])
    rmse_scores = [
        results['regression']['Linear Regression']['RMSE'],
        results['regression']['Random Forest']['RMSE']
    ]
    bars = ax2.bar(models, rmse_scores, color=colors, edgecolor='black')
    ax2.set_ylabel('RMSE')
    ax2.set_title('RMSE Comparison (lower is better)', fontweight='bold')
    for bar, val in zip(bars, rmse_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', fontsize=12, fontweight='bold')
    

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(results['data']['y_cont_test'], results['data']['lr_pred'],
                alpha=0.3, s=10, c=COLORS['primary'])
    ax3.plot([2, 10], [2, 10], 'r--', linewidth=2, label='Perfect fit')
    ax3.set_xlabel('Actual Score')
    ax3.set_ylabel('Predicted Score')
    ax3.set_title('Linear Regression: Actual vs Predicted', fontweight='bold')
    ax3.legend()
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(results['data']['y_cont_test'], results['data']['rf_reg_pred'],
                alpha=0.3, s=10, c=COLORS['success'])
    ax4.plot([2, 10], [2, 10], 'r--', linewidth=2, label='Perfect fit')
    ax4.set_xlabel('Actual Score')
    ax4.set_ylabel('Predicted Score')
    ax4.set_title('Random Forest: Actual vs Predicted', fontweight='bold')
    ax4.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close()
    
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Model Comparison: Classification (Predicting Highly Rated)', 
                 fontsize=18, fontweight='bold', y=0.96)
    
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    models = ['Logistic\nRegression', 'Random\nForest']
    auc_scores = [
        results['classification']['Logistic Regression']['ROC AUC'],
        results['classification']['Random Forest']['ROC AUC']
    ]
    colors = [COLORS['primary'], COLORS['success']]
    bars = ax1.bar(models, auc_scores, color=colors, edgecolor='black')
    ax1.set_ylabel('ROC AUC')
    ax1.set_title('ROC AUC Comparison', fontweight='bold')
    ax1.set_ylim(0, 1)
    for bar, val in zip(bars, auc_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', fontsize=12, fontweight='bold')
    
    ax2 = fig.add_subplot(gs[0, 1])
    f1_scores = [
        results['classification']['Logistic Regression']['F1'],
        results['classification']['Random Forest']['F1']
    ]
    bars = ax2.bar(models, f1_scores, color=colors, edgecolor='black')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score Comparison', fontweight='bold')
    ax2.set_ylim(0, max(f1_scores) * 1.3 if max(f1_scores) > 0 else 0.5)
    for bar, val in zip(bars, f1_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', fontsize=12, fontweight='bold')
    
    ax3 = fig.add_subplot(gs[1, :])
    
    fpr_log, tpr_log, _ = roc_curve(results['data']['y_bin_test'], results['data']['log_prob'])
    ax3.plot(fpr_log, tpr_log, color=COLORS['primary'], lw=2,
             label=f'Logistic Regression (AUC = {auc_scores[0]:.3f})')
    
    fpr_rf, tpr_rf, _ = roc_curve(results['data']['y_bin_test'], results['data']['rf_prob'])
    ax3.plot(fpr_rf, tpr_rf, color=COLORS['success'], lw=2,
             label=f'Random Forest (AUC = {auc_scores[1]:.3f})')
    
    ax3.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC Curve Comparison', fontweight='bold')
    ax3.legend(loc='lower right')
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1.05])
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close()
    
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Feature Importance: Random Forest', fontsize=18, fontweight='bold', y=0.96)
    
    gs = GridSpec(1, 2, figure=fig, wspace=0.4)
    
    ax1 = fig.add_subplot(gs[0, 0])
    top_reg = rf_reg_importance.head(15)
    ax1.barh(range(len(top_reg)), top_reg['Importance'].values[::-1],
             color=COLORS['primary'], edgecolor='black')
    ax1.set_yticks(range(len(top_reg)))
    ax1.set_yticklabels([x[:25] for x in top_reg['Feature'].values[::-1]], fontsize=8)
    ax1.set_xlabel('Importance')
    ax1.set_title('Regression (Score)', fontweight='bold')
    
    ax2 = fig.add_subplot(gs[0, 1])
    top_clf = rf_clf_importance.head(15)
    ax2.barh(range(len(top_clf)), top_clf['Importance'].values[::-1],
             color=COLORS['success'], edgecolor='black')
    ax2.set_yticks(range(len(top_clf)))
    ax2.set_yticklabels([x[:25] for x in top_clf['Feature'].values[::-1]], fontsize=8)
    ax2.set_xlabel('Importance')
    ax2.set_title('Classification (Highly Rated)', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close()
    
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Final Model Comparison Summary', fontsize=20, fontweight='bold', y=0.96)
    
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    summary_text = f"""
    ══════════════════════════════════════════════════════════════
                         REGRESSION MODELS
                      (Predicting Score 1-10)
    ══════════════════════════════════════════════════════════════
    
    Model                      R²          RMSE
    ─────────────────────────────────────────────────────────────
    Linear Regression         {results['regression']['Linear Regression']['R²']:.4f}      {results['regression']['Linear Regression']['RMSE']:.4f}
    Random Forest             {results['regression']['Random Forest']['R²']:.4f}      {results['regression']['Random Forest']['RMSE']:.4f}
    
    Winner: {'Random Forest' if results['regression']['Random Forest']['R²'] > results['regression']['Linear Regression']['R²'] else 'Linear Regression'} (higher R², lower RMSE)
    
    
    ══════════════════════════════════════════════════════════════
                       CLASSIFICATION MODELS
                    (Predicting Highly Rated ≥8.0)
    ══════════════════════════════════════════════════════════════
    
    Model                      ROC AUC     F1 Score
    ─────────────────────────────────────────────────────────────
    Logistic Regression       {results['classification']['Logistic Regression']['ROC AUC']:.4f}      {results['classification']['Logistic Regression']['F1']:.4f}
    Random Forest             {results['classification']['Random Forest']['ROC AUC']:.4f}      {results['classification']['Random Forest']['F1']:.4f}
    
    Winner: {'Random Forest' if results['classification']['Random Forest']['ROC AUC'] > results['classification']['Logistic Regression']['ROC AUC'] else 'Logistic Regression'} (higher ROC AUC)
    
    
    ══════════════════════════════════════════════════════════════
                           KEY FINDINGS
    ══════════════════════════════════════════════════════════════
    
    1. Random Forest handles sparse categorical features well
       (as suggested by TA feedback)
    
    2. Both model types achieve similar performance, suggesting
       the relationships are approximately linear
    
    3. Top predictors across all models:
       - Year (more recent = higher scores)
       - Episodes (longer series tend to score higher)
       - Drama genre (consistently positive effect)
       - Source material (Manga > Original)
    
    4. Class imbalance (only 3.9% highly rated) affects
       classification metrics - ROC AUC is most reliable
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8fafc', edgecolor='gray'))
    
    pdf.savefig(fig)
    plt.close()
    
    pdf.close()
    print(f"✓ Saved: {output_path}")


def print_final_summary(results):
    """Print final summary of all models."""
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    print("""
┌─────────────────────────────────────────────────────────────┐
│                    REGRESSION MODELS                        │
│                  (Predicting Score 1-10)                    │
├─────────────────────────────────────────────────────────────┤""")
    print(f"│  Linear Regression    R² = {results['regression']['Linear Regression']['R²']:.3f}    RMSE = {results['regression']['Linear Regression']['RMSE']:.3f}   │")
    print(f"│  Random Forest        R² = {results['regression']['Random Forest']['R²']:.3f}    RMSE = {results['regression']['Random Forest']['RMSE']:.3f}   │")
    print("""├─────────────────────────────────────────────────────────────┤
│                  CLASSIFICATION MODELS                      │
│               (Predicting Highly Rated ≥8.0)                │
├─────────────────────────────────────────────────────────────┤""")
    print(f"│  Logistic Regression  AUC = {results['classification']['Logistic Regression']['ROC AUC']:.3f}    F1 = {results['classification']['Logistic Regression']['F1']:.3f}    │")
    print(f"│  Random Forest        AUC = {results['classification']['Random Forest']['ROC AUC']:.3f}    F1 = {results['classification']['Random Forest']['F1']:.3f}    │")
    print("""└─────────────────────────────────────────────────────────────┘
    """)
    
    reg_winner = 'Random Forest' if results['regression']['Random Forest']['R²'] > results['regression']['Linear Regression']['R²'] else 'Linear Regression'
    clf_winner = 'Random Forest' if results['classification']['Random Forest']['ROC AUC'] > results['classification']['Logistic Regression']['ROC AUC'] else 'Logistic Regression'
    
    print(f"REGRESSION WINNER:      {reg_winner}")
    print(f"CLASSIFICATION WINNER:  {clf_winner}")


def main():
    
    DATA_PATH = "data/anime_modeling_data.csv"
    OUTPUT_PATH = "outputs/model_comparison.pdf"
    
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data not found at {DATA_PATH}")
        print("Please run 01_data_preparation.py first")
        return
    
    print("="*60)
    print("RANDOM FOREST & MODEL COMPARISON")
    print("="*60)
    print(f"\nLoading data from: {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    print(f"Sample size: {len(df):,} anime")
    
    X, y_continuous, y_binary, feature_names = prepare_features(df)
    
    rf_reg, X_test_reg, y_test_reg, y_pred_reg, rf_reg_importance = \
        random_forest_regression(X, y_continuous, feature_names)
    
    rf_clf, X_test_clf, y_test_clf, y_pred_clf, y_prob_clf, rf_clf_importance = \
        random_forest_classification(X, y_binary, feature_names)
    
    results = compare_all_models(X, y_continuous, y_binary, feature_names)
    
    create_comparison_plots(results, rf_reg_importance, rf_clf_importance, OUTPUT_PATH)
    
    print_final_summary(results)
    
    print("\n✓ All outputs saved to outputs/ folder")
    
    return results, rf_reg_importance, rf_clf_importance


if __name__ == "__main__":
    results, rf_reg_imp, rf_clf_imp = main()
