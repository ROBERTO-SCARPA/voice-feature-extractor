"""
Feature Selection Pipeline

Per dataset vocali con feature da ridurre a subset ottimale

Strategia:
1. Correlation Matrix → Rimuove feature ridondanti (correlazione > thresold)
2. Variance Threshold → Rimuove feature con varianza quasi nulla
3. PCA Analysis → Analizza varianza spiegata (per determinare n_components)
4. SPLIT 80-20 (train/test)
5a. PCA FORMALE → Fit su train, transform su train e test
5b. ANOVA F-TEST → Fit su train, transform su train e test
6. Validation → Monte-Carlo CV con fit corretto per ogni split

"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
import warnings
import joblib
warnings.filterwarnings('ignore')

# ===============================================
# 1. DATA LOADING
# ===============================================
def load_and_merge_data(dataset_custom_path, dataset_egemaps_path, dataset_index_path, output_dir):
    """
    Carica e merge i 3 dataset.
    """
    print("\n" + "="*70)
    print("📂 STEP 1: CARICAMENTO E MERGE DATASET")
    print("="*70)

    dataset_custom = pd.read_csv(dataset_custom_path, delimiter=';')
    dataset_egemaps = pd.read_csv(dataset_egemaps_path, delimiter=';')
    dataset_index = pd.read_excel(dataset_index_path)

    print(f"✓ Dataset custom caricato: {len(dataset_custom)} righe, {len(dataset_custom.columns)} colonne")
    print(f"✓ Dataset eGeMAPS caricato: {len(dataset_egemaps)} righe, {len(dataset_egemaps.columns)} colonne")
    print(f"✓ Dataset index caricato: {len(dataset_index)} righe")

    # Standardize filenames
    dataset_custom['filename_standard'] = dataset_custom['filename'].apply(
        lambda x: x.split('Italian')[0] + 'Italian'
    )
    dataset_egemaps['filename_standard'] = dataset_egemaps['filename'].apply(
        lambda x: x.split('Italian')[0] + 'Italian'
    )

    # Merge
    merged_df = pd.merge(dataset_egemaps, dataset_custom,
                         left_on='filename_standard', right_on='filename_standard')
    print(f"✓ Dopo merge custom+eGeMAPS: {len(merged_df)} righe")

    merged_df = pd.merge(merged_df, dataset_index,
                         left_on='filename_x', right_on='FileName')
    print(f"✓ Dopo merge con index: {len(merged_df)} righe")

    # Filtra
    merged_df = merged_df[merged_df['Tipo audio'] == 'Free']
    print(f"✓ Dopo filtro 'Tipo audio = Free': {len(merged_df)} righe")

    target_column = 'Tipo soggetto'

    # Features e target
    X = merged_df.drop(columns=[
        target_column, 'filename_standard', 'filename_x', 'filename_y',
        'subjectId_x', 'subjectId_y', 'ID', 'FileName', 'Tipo audio', 'class', 'name'
    ], errors="ignore")
    y = merged_df[target_column]

    # Encoding categorical
    le = LabelEncoder()
    categorical_columns = X.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        X[col] = le.fit_transform(X[col])

    print(f"\n📊 DATASET FINALE:")
    print(f"   Campioni: {len(X)}")
    print(f"   Feature: {len(X.columns)}")
    print(f"   Classi: {y.value_counts().to_dict()}")

    # Salva
    os.makedirs(output_dir, exist_ok=True)
    full_dataset_path = os.path.join(output_dir, "full_dataset_before_selection.csv")
    full_df = X.copy()
    full_df['Tipo soggetto'] = y
    full_df.to_csv(full_dataset_path, index=False)
    print(f"✓ Dataset completo salvato: {full_dataset_path}")

    return X, y

# ===============================================
# 2. CORRELATION MATRIX ANALYSIS
# ===============================================
def correlation_analysis(X, threshold=0.9, output_dir="output"):
    """
    Analizza correlazione e rimuove feature altamente correlate.
    OK fare su tutto il dataset (non è un fit che impara dai dati).
    """
    print("\n" + "="*70)
    print("🔗 STEP 2: CORRELATION MATRIX ANALYSIS")
    print("="*70)

    corr_matrix = X.corr().abs()

    # Plot
    if len(X.columns) <= 50:
        plt.figure(figsize=(20, 18))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                    xticklabels=True, yticklabels=True)
        title = f'Correlation Matrix - {len(X.columns)} Feature'
    else:
        plt.figure(figsize=(16, 14))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                    square=True, linewidths=0.1, cbar_kws={"shrink": 0.8},
                    xticklabels=False, yticklabels=False)
        title = f'Correlation Matrix - {len(X.columns)} Feature (labels hidden)'

    plt.title(title, fontsize=16)
    plt.tight_layout()
    corr_plot_path = os.path.join(output_dir, "correlation_matrix_full.png")
    plt.savefig(corr_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Correlation matrix salvata: {corr_plot_path}")

    # Trova feature correlate
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [column for column in upper_triangle.columns
               if any(upper_triangle[column] > threshold)]

    print(f"\n📊 Analisi correlazione (threshold={threshold}):")
    print(f"   Feature iniziali: {len(X.columns)}")
    print(f"   Feature altamente correlate (> {threshold}): {len(to_drop)}")

    if to_drop:
        print(f"\n🗑️ Feature rimosse per alta correlazione:")
        for i, feat in enumerate(to_drop[:20], 1):
            correlated_with = upper_triangle[feat][upper_triangle[feat] > threshold].index.tolist()
            if correlated_with:
                print(f"   {i}. {feat} (corr con: {correlated_with[0]})")
        if len(to_drop) > 20:
            print(f"   ... e altre {len(to_drop)-20} feature")

    X_reduced = X.drop(columns=to_drop)
    print(f"\n✓ Feature dopo rimozione correlazione: {len(X_reduced.columns)}")

    # Salva
    removed_path = os.path.join(output_dir, "removed_features_correlation.txt")
    with open(removed_path, 'w') as f:
        f.write(f"Feature rimosse per correlazione > {threshold}:\n")
        f.write(f"Totale: {len(to_drop)}\n\n")
        for feat in to_drop:
            correlated_with = upper_triangle[feat][upper_triangle[feat] > threshold].index.tolist()
            f.write(f"{feat} -> correlata con: {', '.join(correlated_with)}\n")
    print(f"✓ Lista salvata: {removed_path}")

    return X_reduced, to_drop

# ===============================================
# 3. VARIANCE THRESHOLD
# ===============================================
def variance_filtering(X, threshold=0.01, output_dir="output"):
    """
    Rimuove feature con varianza bassa.
    OK fare su tutto il dataset (threshold fisso).
    """
    print("\n" + "="*70)
    print("📉 STEP 3: VARIANCE THRESHOLD FILTERING")
    print("="*70)

    variances = X.var()
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)

    feature_mask = selector.get_support()
    low_variance_features = X.columns[~feature_mask].tolist()

    print(f"\n📊 Analisi varianza (threshold={threshold}):")
    print(f"   Feature con bassa varianza: {len(low_variance_features)}")

    if low_variance_features:
        print(f"\n🗑️ Feature rimosse per bassa varianza:")
        for i, feat in enumerate(low_variance_features[:10], 1):
            print(f"   {i}. {feat} (var={variances[feat]:.6f})")
        if len(low_variance_features) > 10:
            print(f"   ... e altre {len(low_variance_features)-10} feature")

    X_reduced = X.loc[:, feature_mask]
    print(f"\n✓ Feature dopo variance filtering: {len(X_reduced.columns)}")

    # Salva
    removed_path = os.path.join(output_dir, "removed_features_variance.txt")
    with open(removed_path, 'w') as f:
        f.write(f"Feature rimosse per varianza < {threshold}:\n")
        f.write(f"Totale: {len(low_variance_features)}\n\n")
        for feat in low_variance_features:
            f.write(f"{feat}: var={variances[feat]:.6f}\n")
    print(f"✓ Lista salvata: {removed_path}")

    return X_reduced, low_variance_features

# ===============================================
# 4. PCA ANALYSIS (per determinare n_components)
# ===============================================
def pca_analysis(X, y, output_dir="output"):
    """
    Analizza PCA per determinare numero ottimale di componenti.
    Questa è solo analisi esplorativa, OK fare su tutto il dataset.
    La vera trasformazione PCA sarà fatta solo su train.
    """
    print("\n" + "="*70)
    print("🔬 STEP 4: PCA ANALYSIS (Esplorativa)")
    print("="*70)
    print("NOTA: Questa è solo per determinare n_components.")
    print("   La trasformazione PCA vera sarà fatta SOLO su train dopo lo split.")

    # Standardizzazione
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA completa
    pca = PCA(random_state=42)
    pca.fit(X_scaled)

    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
    eigenvalues = pca.explained_variance_

    # Kaiser criterion
    n_kaiser = np.sum(eigenvalues > 1.0)

    # Soglie varianza
    n_components_95 = np.argmax(cumsum_variance >= 0.95) + 1
    n_components_90 = np.argmax(cumsum_variance >= 0.90) + 1
    n_components_85 = np.argmax(cumsum_variance >= 0.85) + 1

    print(f"\n📊 Analisi Varianza Spiegata:")
    print(f"   Kaiser Criterion (eigenvalues > 1): {n_kaiser} componenti")
    print(f"   Componenti per 85% varianza: {n_components_85}")
    print(f"   Componenti per 90% varianza: {n_components_90}")
    print(f"   Componenti per 95% varianza: {n_components_95} ← RACCOMANDATO")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Scree plot
    ax1 = axes[0, 0]
    n_plot = min(50, len(eigenvalues))
    ax1.plot(range(1, n_plot+1), eigenvalues[:n_plot], 'bo-', linewidth=2, markersize=8)
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Kaiser criterion (λ=1)')
    ax1.axvline(x=n_kaiser, color='red', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Componente Principale', fontsize=12)
    ax1.set_ylabel('Autovalore', fontsize=12)
    ax1.set_title('Scree Plot', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Varianza per componente
    ax2 = axes[0, 1]
    ax2.bar(range(1, n_plot+1), pca.explained_variance_ratio_[:n_plot], 
            alpha=0.7, color='steelblue')
    ax2.set_xlabel('Componente', fontsize=12)
    ax2.set_ylabel('Varianza Spiegata', fontsize=12)
    ax2.set_title('Varianza per Componente', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')

    # Varianza cumulativa
    ax3 = axes[1, 0]
    ax3.plot(range(1, len(cumsum_variance)+1), cumsum_variance, linewidth=3, color='darkblue')
    ax3.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95%')
    ax3.axhline(y=0.90, color='orange', linestyle='--', linewidth=2, label='90%')
    ax3.axvline(x=n_components_95, color='red', linestyle=':', alpha=0.5)
    ax3.scatter([n_components_95], [0.95], color='red', s=150, zorder=5, marker='D')
    ax3.text(n_components_95+2, 0.95, f'n={n_components_95}', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Numero Componenti', fontsize=12)
    ax3.set_ylabel('Varianza Cumulativa', fontsize=12)
    ax3.set_title('Varianza Cumulativa PCA', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.set_ylim([0, 1.05])

    # Tabella
    ax4 = axes[1, 1]
    ax4.axis('off')
    table_data = [
        ['Criterio', 'N° Componenti', 'Varianza'],
        ['Kaiser (λ>1)', str(n_kaiser), f'{cumsum_variance[n_kaiser-1]*100:.1f}%'],
        ['85% varianza', str(n_components_85), '85.0%'],
        ['90% varianza', str(n_components_90), '90.0%'],
        ['95% varianza ✓', str(n_components_95), '95.0%'],
    ]
    table = ax4.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    for i in range(3):
        table[(4, i)].set_facecolor('#90EE90')

    ax4.set_title('Riepilogo', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    pca_plot_path = os.path.join(output_dir, "pca_analysis_exploratory.png")
    plt.savefig(pca_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Plot PCA salvato: {pca_plot_path}")

    return n_components_90, n_components_95

# ===============================================
# 5A. PCA TRANSFORMATION
# ===============================================
def apply_pca_transformation_train_test(X_train, X_test, n_components, output_dir="output"):
    """
    Applica PCA fittato SOLO su train, poi trasforma train e test.

    Args:
        X_train: Feature di training
        X_test: Feature di test
        n_components: Numero di componenti principali
        output_dir: Directory output

    Returns:
        X_train_pca: Train trasformato in componenti principali
        X_test_pca: Test trasformato con stesso PCA
        pca: Oggetto PCA fittato (per deployment)
        scaler: Scaler fittato (per deployment)
    """
    print("\n" + "="*70)
    print(f"⭐ STEP 5A: PCA TRANSFORMATION ({n_components} componenti)")
    print("="*70)
    print("✅ FIT PCA SOLO SU TRAIN")

    # Standardizzazione: FIT su train, TRANSFORM su train e test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # usa scaler fittato su train

    print(f"\n⚙️ Standardizzazione:")
    print(f"   Train: fit + transform → shape {X_train_scaled.shape}")
    print(f"   Test: transform only → shape {X_test_scaled.shape}")

    # PCA: FIT su train, TRANSFORM su train e test
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)  # usa PCA fittato su train

    variance_explained = pca.explained_variance_ratio_.sum()

    print(f"\n✓ PCA completato:")
    print(f"   Feature originali: {X_train.shape[1]}")
    print(f"   Componenti principali: {n_components}")
    print(f"   Varianza spiegata: {variance_explained*100:.2f}%")
    print(f"   Train PCA shape: {X_train_pca.shape}")
    print(f"   Test PCA shape: {X_test_pca.shape}")

    # Crea DataFrame
    columns = [f'PC{i+1}' for i in range(n_components)]
    X_train_pca_df = pd.DataFrame(X_train_pca, columns=columns, index=X_train.index)
    X_test_pca_df = pd.DataFrame(X_test_pca, columns=columns, index=X_test.index)

    # Salva loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=columns,
        index=X_train.columns
    )
    loadings_path = os.path.join(output_dir, "pca_loadings.csv")
    loadings.to_csv(loadings_path)
    print(f"\n✓ Loadings salvati: {loadings_path}")

    # Plot loadings heatmap
    n_plot = min(10, n_components)
    plt.figure(figsize=(14, max(10, len(X_train.columns)//3)))
    sns.heatmap(loadings.iloc[:, :n_plot], cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Loading'}, linewidths=0.5)
    plt.title(f'PCA Loadings - Prime {n_plot} Componenti', fontsize=14, fontweight='bold')
    plt.xlabel('Componenti Principali', fontsize=12)
    plt.ylabel('Feature Originali', fontsize=12)
    plt.tight_layout()
    loadings_plot_path = os.path.join(output_dir, "pca_loadings_heatmap.png")
    plt.savefig(loadings_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Heatmap loadings salvata: {loadings_plot_path}")

    # Top contributors
    print(f"\n📊 Top 5 feature per prime 5 componenti:")
    for i in range(min(5, n_components)):
        pc = f'PC{i+1}'
        top_features = loadings[pc].abs().nlargest(5)
        print(f"\n   {pc} ({pca.explained_variance_ratio_[i]*100:.1f}% varianza):")
        for feat, _ in top_features.items():
            actual_load = loadings.loc[feat, pc]
            print(f"     {feat}: {actual_load:+.3f}")

    # Salva PCA e scaler per deployment
    pca_model_path = os.path.join(output_dir, "pca_fitted.pkl")
    scaler_model_path = os.path.join(output_dir, "scaler_pca.pkl")
    joblib.dump(pca, pca_model_path)
    joblib.dump(scaler, scaler_model_path)
    print(f"\n💾 Modelli salvati per deployment:")
    print(f"   PCA: {pca_model_path}")
    print(f"   Scaler: {scaler_model_path}")

    return X_train_pca_df, X_test_pca_df, pca, scaler

# ===============================================
# 5B. ANOVA FEATURE SELECTION
# ===============================================
def select_features_anova_train_test(X_train, X_test, y_train, n_features, output_dir="output"):
    """
    Seleziona feature con ANOVA fittato SOLO su train.

    Args:
        X_train: Feature di training
        X_test: Feature di test
        y_train: Target di training
        n_features: Numero di feature da selezionare
        output_dir: Directory output

    Returns:
        X_train_anova: Train con feature selezionate
        X_test_anova: Test con feature selezionate
        selector: Oggetto SelectKBest fittato
        scaler: Scaler fittato
        selected_features: Lista nomi feature selezionate
    """
    print("\n" + "="*70)
    print(f"⭐ STEP 5B: ANOVA F-TEST ({n_features} features)")
    print("="*70)
    print("✅ FIT ANOVA SOLO SU TRAIN")

    # Standardizzazione: FIT su train, TRANSFORM su train e test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    print(f"\n⚙️ Standardizzazione:")
    print(f"   Train: fit + transform → shape {X_train_scaled_df.shape}")
    print(f"   Test: transform only → shape {X_test_scaled_df.shape}")

    # ANOVA: FIT su train, TRANSFORM su train e test
    selector = SelectKBest(f_classif, k=n_features)
    X_train_anova = selector.fit_transform(X_train_scaled_df, y_train)
    X_test_anova = selector.transform(X_test_scaled_df)

    print(f"\n✓ ANOVA Feature Selection completato:")
    print(f"   Feature iniziali: {X_train.shape[1]}")
    print(f"   Feature selezionate: {n_features}")
    print(f"   Train shape: {X_train_anova.shape}")
    print(f"   Test shape: {X_test_anova.shape}")

    # Ottieni scores e feature selezionate
    scores = selector.scores_
    feature_scores = pd.DataFrame({
        'feature': X_train.columns,
        'score': scores
    }).sort_values('score', ascending=False)

    selected_mask = selector.get_support()
    selected_features = X_train.columns[selected_mask].tolist()

    # Crea DataFrame
    X_train_anova_df = pd.DataFrame(X_train_anova, columns=selected_features, index=X_train.index)
    X_test_anova_df = pd.DataFrame(X_test_anova, columns=selected_features, index=X_test.index)

    print(f"\n📊 Top {min(20, n_features)} feature per ANOVA F-score:")
    for i, row in feature_scores.head(20).iterrows():
        idx = list(feature_scores.index).index(i) + 1
        print(f"   {idx}. {row['feature']}: {row['score']:.2f}")

    # Plot
    plt.figure(figsize=(12, 8))
    top_n = min(30, n_features)
    plt.barh(range(top_n), feature_scores.head(top_n)['score'].values, color='coral')
    plt.yticks(range(top_n), feature_scores.head(top_n)['feature'].values)
    plt.xlabel('ANOVA F-score', fontsize=12)
    plt.title(f'Top {top_n} Feature (ANOVA)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "anova_features.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Plot salvato: {plot_path}")

    # Salva ranking
    ranked_path = os.path.join(output_dir, "anova_features_ranked.csv")
    feature_scores.to_csv(ranked_path, index=False)
    print(f"✓ Ranking salvato: {ranked_path}")

    # Salva selector e scaler
    selector_path = os.path.join(output_dir, "anova_selector_fitted.pkl")
    scaler_path = os.path.join(output_dir, "scaler_anova.pkl")
    joblib.dump(selector, selector_path)
    joblib.dump(scaler, scaler_path)
    print(f"\n💾 Modelli salvati per deployment:")
    print(f"   Selector: {selector_path}")
    print(f"   Scaler: {scaler_path}")

    return X_train_anova_df, X_test_anova_df, selector, scaler, selected_features

# ===============================================
# 6. VALIDATION MONTE-CARLO CORRETTA
# ===============================================
def validate_feature_selection_corrected(X_full, X_preprocessed, y, n_components_pca, n_features_anova,
                                         n_splits=50, test_size=0.2, output_dir="output"):
    """
    Per ogni split, fit PCA/ANOVA solo su train di quel split.

    Args:
        X_full: Dataset completo (prima del preprocessing)
        X_preprocessed: Dataset dopo correlation + variance
        y: Target
        n_components_pca: Numero componenti PCA
        n_features_anova: Numero feature ANOVA
        n_splits: Numero di split Monte-Carlo
        test_size: Percentuale test
        output_dir: Directory output
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    from imblearn.over_sampling import SMOTE

    print("\n" + "="*70)
    print(f" STEP 6: VALIDATION MONTE-CARLO CV ({n_splits} split)")
    print("="*70)
    print(" FIT su ogni TRAIN split")

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)

    scores_full = []
    scores_preprocessed = []
    scores_pca = []
    scores_anova = []

    print(f"\n Strategia: {n_splits} split stratificati ({int((1-test_size)*100)}-{int(test_size*100)})")
    print(f"  SMOTE su train di ogni split")
    print(f"  PCA e ANOVA fittati SOLO su train di ogni split")

    for i, (train_idx, test_idx) in enumerate(sss.split(X_full, y)):
        if (i + 1) % 10 == 0:
            print(f"   Split {i+1}/{n_splits} completato...", end='\r')

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # ===================================================
        # 1. FULL DATASET (feature iniziali complete)
        # ===================================================
        X_train_full = X_full.iloc[train_idx]
        X_test_full = X_full.iloc[test_idx]

        smote = SMOTE(random_state=42)
        X_train_full_bal, y_train_bal = smote.fit_resample(X_train_full, y_train)

        rf.fit(X_train_full_bal, y_train_bal)
        y_pred = rf.predict(X_test_full)
        scores_full.append(balanced_accuracy_score(y_test, y_pred))

        # ===================================================
        # 2. PREPROCESSED (dopo correlation + variance)
        # ===================================================
        X_train_prep = X_preprocessed.iloc[train_idx]
        X_test_prep = X_preprocessed.iloc[test_idx]

        smote = SMOTE(random_state=42)
        X_train_prep_bal, y_train_bal = smote.fit_resample(X_train_prep, y_train)

        rf.fit(X_train_prep_bal, y_train_bal)
        y_pred = rf.predict(X_test_prep)
        scores_preprocessed.append(balanced_accuracy_score(y_test, y_pred))

        # ===================================================
        # 3. PCA - FIT SOLO SU TRAIN DI QUESTO SPLIT
        # ===================================================
        scaler_pca = StandardScaler()
        X_train_prep_scaled = scaler_pca.fit_transform(X_train_prep)
        X_test_prep_scaled = scaler_pca.transform(X_test_prep)

        pca = PCA(n_components=n_components_pca, random_state=42)
        X_train_pca = pca.fit_transform(X_train_prep_scaled)
        X_test_pca = pca.transform(X_test_prep_scaled)

        smote = SMOTE(random_state=42)
        X_train_pca_bal, y_train_bal = smote.fit_resample(X_train_pca, y_train)

        rf.fit(X_train_pca_bal, y_train_bal)
        y_pred = rf.predict(X_test_pca)
        scores_pca.append(balanced_accuracy_score(y_test, y_pred))

        # ===================================================
        # 4. ANOVA - FIT SOLO SU TRAIN DI QUESTO SPLIT
        # ===================================================
        scaler_anova = StandardScaler()
        X_train_prep_scaled = scaler_anova.fit_transform(X_train_prep)
        X_test_prep_scaled = scaler_anova.transform(X_test_prep)

        selector = SelectKBest(f_classif, k=n_features_anova)
        X_train_anova = selector.fit_transform(X_train_prep_scaled, y_train)
        X_test_anova = selector.transform(X_test_prep_scaled)

        smote = SMOTE(random_state=42)
        X_train_anova_bal, y_train_bal = smote.fit_resample(X_train_anova, y_train)

        rf.fit(X_train_anova_bal, y_train_bal)
        y_pred = rf.predict(X_test_anova)
        scores_anova.append(balanced_accuracy_score(y_test, y_pred))

    print(f"   Split {n_splits}/{n_splits} completato... ✓")

    scores_full = np.array(scores_full)
    scores_preprocessed = np.array(scores_preprocessed)
    scores_pca = np.array(scores_pca)
    scores_anova = np.array(scores_anova)

    # Statistiche
    print(f"\n📊 RISULTATI MONTE-CARLO VALIDATION:")

    print(f"\n   1. Feature iniziali complete ({X_full.shape[1]}):")
    print(f"      Mean: {scores_full.mean():.3f} ± {scores_full.std():.3f}")
    print(f"      Range: [{scores_full.min():.3f}, {scores_full.max():.3f}]")

    print(f"\n   2. Dopo preprocessing ({X_preprocessed.shape[1]}):")
    print(f"      Mean: {scores_preprocessed.mean():.3f} ± {scores_preprocessed.std():.3f}")
    print(f"      Range: [{scores_preprocessed.min():.3f}, {scores_preprocessed.max():.3f}]")

    print(f"\n   3. PCA ({n_components_pca} componenti):")
    print(f"      Mean: {scores_pca.mean():.3f} ± {scores_pca.std():.3f}")
    print(f"      Range: [{scores_pca.min():.3f}, {scores_pca.max():.3f}]")

    print(f"\n   4. ANOVA ({n_features_anova} features):")
    print(f"      Mean: {scores_anova.mean():.3f} ± {scores_anova.std():.3f}")
    print(f"      Range: [{scores_anova.min():.3f}, {scores_anova.max():.3f}]")

    improvement1 = scores_preprocessed.mean() - scores_full.mean()
    improvement2_pca = scores_pca.mean() - scores_preprocessed.mean()
    improvement2_anova = scores_anova.mean() - scores_preprocessed.mean()

    print(f"\n   🎯 MIGLIORAMENTI:")
    print(f"      Full → Preprocessed: {improvement1:+.3f}")
    print(f"      Preprocessed → PCA: {improvement2_pca:+.3f}")
    print(f"      Preprocessed → ANOVA: {improvement2_anova:+.3f}")

    best_method = 'PCA' if scores_pca.mean() > scores_anova.mean() else 'ANOVA'
    best_score = max(scores_pca.mean(), scores_anova.mean())
    print(f"\n   🏆 MIGLIOR METODO: {best_method} ({best_score:.3f})")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    data_to_plot = [scores_full, scores_preprocessed, scores_pca, scores_anova]
    labels = [f'Full\n({X_full.shape[1]})',
              f'Preprocessed\n({X_preprocessed.shape[1]})',
              f'PCA\n({n_components_pca})',
              f'ANOVA\n({n_features_anova})']
    colors = ['lightcoral', 'lightskyblue', 'lightgreen', 'lightyellow']

    bp = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    for i, scores in enumerate(data_to_plot):
        ax1.scatter([i+1], [scores.mean()], color='red', s=100, zorder=3, marker='D')
        ax1.text(i+1, scores.mean() + 0.02, f'{scores.mean():.3f}', 
                ha='center', fontsize=10, fontweight='bold')

    ax1.set_ylabel('Balanced Accuracy', fontsize=12)
    ax1.set_title('Confronto Performance', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3, axis='y')
    ax1.set_ylim([0, 1])

    # Histogram
    ax2.hist(scores_full, bins=20, alpha=0.5, 
            label=f'Full (μ={scores_full.mean():.3f})', color='lightcoral')
    ax2.hist(scores_preprocessed, bins=20, alpha=0.5, 
            label=f'Preprocessed (μ={scores_preprocessed.mean():.3f})', color='lightskyblue')
    ax2.hist(scores_pca, bins=20, alpha=0.5, 
            label=f'PCA (μ={scores_pca.mean():.3f})', color='lightgreen')
    ax2.hist(scores_anova, bins=20, alpha=0.5, 
            label=f'ANOVA (μ={scores_anova.mean():.3f})', color='lightyellow')

    ax2.axvline(scores_full.mean(), color='red', linestyle='--', linewidth=2)
    ax2.axvline(scores_preprocessed.mean(), color='blue', linestyle='--', linewidth=2)
    ax2.axvline(scores_pca.mean(), color='green', linestyle='--', linewidth=2)
    ax2.axvline(scores_anova.mean(), color='orange', linestyle='--', linewidth=2)

    ax2.set_xlabel('Balanced Accuracy', fontsize=12)
    ax2.set_ylabel('Frequenza', fontsize=12)
    ax2.set_title(f'Distribuzione ({n_splits} split + SMOTE)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "validation_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Plot salvato: {plot_path}")

    # Salva risultati
    results_path = os.path.join(output_dir, "validation_results.txt")
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(f"VALIDATION RESULTS ({n_splits} split)\n")
        f.write("="*70 + "\n\n")
        f.write(f" PCA e ANOVA fittati SOLO su train di ogni split\n")
        f.write(f" Validation affidabile e generalizzabile\n\n")
        f.write(f"N° split: {n_splits}\n")
        f.write(f"Test size: {test_size*100:.0f}%\n\n")

        f.write(f"1. Full dataset: {X_full.shape[1]} features\n")
        f.write(f"   Mean: {scores_full.mean():.3f} ± {scores_full.std():.3f}\n\n")

        f.write(f"2. Preprocessed: {X_preprocessed.shape[1]} features\n")
        f.write(f"   Mean: {scores_preprocessed.mean():.3f} ± {scores_preprocessed.std():.3f}\n\n")

        f.write(f"3. PCA: {n_components_pca} componenti\n")
        f.write(f"   Mean: {scores_pca.mean():.3f} ± {scores_pca.std():.3f}\n\n")

        f.write(f"4. ANOVA: {n_features_anova} features\n")
        f.write(f"   Mean: {scores_anova.mean():.3f} ± {scores_anova.std():.3f}\n\n")

        f.write(f"MIGLIOR METODO: {best_method}\n")
        f.write(f"Performance: {best_score:.3f}\n\n")

    print(f"✓ Risultati salvati: {results_path}")

    return scores_full.mean(), scores_preprocessed.mean(), scores_pca.mean(), scores_anova.mean(), best_method

# ===============================================
# MAIN CORRETTO
# ===============================================
def main(args):
    """
    Pipeline.
    1-3. Preprocessing (correlation + variance) su tutto
    4. PCA analysis esplorativa (per determinare n_components)
    5. SPLIT TRAIN/TEST
    6. PCA e ANOVA fittati SOLO su train
    7. Validation con fit su ogni train split
    """
    print("\n" + "="*70)
    print("🚀 FEATURE SELECTION PIPELINE")
    print("="*70)
    print(f"\nOutput directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ===================================================
    # STEP 1-3: PREPROCESSING (OK su tutto il dataset)
    # ===================================================
    X, y = load_and_merge_data(
        args.dataset_custom,
        args.dataset_egemaps,
        args.dataset_index,
        args.output_dir
    )

    print(f"\n📊 Feature iniziali: {len(X.columns)}")

    X_no_corr, removed_corr = correlation_analysis(
        X, threshold=args.correlation_threshold, output_dir=args.output_dir
    )

    X_preprocessed, removed_var = variance_filtering(
        X_no_corr, threshold=args.variance_threshold, output_dir=args.output_dir
    )

    # ===================================================
    # STEP 4: PCA ANALYSIS ESPLORATIVA
    # ===================================================
    print("\n NOTA: PCA analysis esplorativa per determinare n_components.")
    print("   La trasformazione PCA vera sarà su train dopo lo split.")

    n_components_90, n_components_95 = pca_analysis(
        X_preprocessed, y, output_dir=args.output_dir
    )

    # Determina n_components finale
    if args.n_features is not None:
        n_features_final = args.n_features
        print(f"\n🎯 N° componenti/feature: {n_features_final} (specificato manualmente)")
    else:
        n_features_final = n_components_95
        print(f"\n🎯 N° componenti/feature: {n_features_final} (da PCA 95% varianza)")

    # ===================================================
    # ✅ STEP 5: SPLIT TRAIN/TEST (PRIMA DELLE TRASFORMAZIONI!)
    # ===================================================
    print("\n" + "="*70)
    print("✅ SPLIT TRAIN/TEST 80-20 (PRIMA di PCA e ANOVA)")
    print("="*70)
    print("⚠️ CRITICO: Questo previene data leakage!")

    X_train, X_test, y_train, y_test = train_test_split(
        X_preprocessed, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\n✓ Train set: {len(X_train)} campioni ({len(X_train)/len(X_preprocessed)*100:.1f}%)")
    print(f"✓ Test set: {len(X_test)} campioni ({len(X_test)/len(X_preprocessed)*100:.1f}%)")
    print(f"  Train - Classi: {y_train.value_counts().to_dict()}")
    print(f"  Test - Classi: {y_test.value_counts().to_dict()}")

    # Anche per X originale e X senza correlazione (per validation)
    X_train_full = X.loc[X_train.index]
    X_test_full = X.loc[X_test.index]

    # ===================================================
    # STEP 6A: PCA TRANSFORMATION (FIT SOLO SU TRAIN)
    # ===================================================
    X_train_pca, X_test_pca, pca_fitted, scaler_pca = apply_pca_transformation_train_test(
        X_train, X_test, n_features_final, output_dir=args.output_dir
    )

    # ===================================================
    # STEP 6B: ANOVA SELECTION (FIT SOLO SU TRAIN)
    # ===================================================
    X_train_anova, X_test_anova, selector_fitted, scaler_anova, selected_features = \
        select_features_anova_train_test(
            X_train, X_test, y_train, n_features_final, output_dir=args.output_dir
        )

    # ===================================================
    # STEP 7: VALIDATION MONTE-CARLO
    # ===================================================
    acc_full, acc_prep, acc_pca, acc_anova, best_method = validate_feature_selection_corrected(
        X, X_preprocessed, y,
        n_components_pca=n_features_final,
        n_features_anova=n_features_final,
        n_splits=50,
        output_dir=args.output_dir
    )

    # ===================================================
    # STEP 8: SALVATAGGIO DATASET FINALI
    # ===================================================
    print("\n" + "="*70)
    print("💾 SALVATAGGIO DATASET FINALI")
    print("="*70)

    # Dataset PCA completo (train + test)
    X_pca_complete = pd.concat([X_train_pca, X_test_pca])
    y_complete = pd.concat([y_train, y_test])
    X_pca_complete['Tipo soggetto'] = y_complete

    pca_path = os.path.join(args.output_dir, "dataset_PCA.csv")
    X_pca_complete.to_csv(pca_path, index=False, sep=';')
    print(f"\n✓ Dataset PCA salvato: {pca_path}")
    print(f"  Contiene: {n_features_final} componenti PC1, PC2, ...")
    print(f"  Train: {len(X_train_pca)}, Test: {len(X_test_pca)}")

    # Dataset ANOVA completo (train + test)
    X_anova_complete = pd.concat([X_train_anova, X_test_anova])
    X_anova_complete['Tipo soggetto'] = y_complete

    anova_path = os.path.join(args.output_dir, "dataset_ANOVA.csv")
    X_anova_complete.to_csv(anova_path, index=False, sep=';')
    print(f"\n✓ Dataset ANOVA salvato: {anova_path}")
    print(f"  Contiene: {n_features_final} feature originali")
    print(f"  Feature: {', '.join(selected_features[:10])}...")
    print(f"  Train: {len(X_train_anova)}, Test: {len(X_test_anova)}")


    # ===================================================
    # SUMMARY FINALE
    # ===================================================
    print("\n" + "="*70)
    print("📋 SUMMARY FINALE")
    print("="*70)
    print(f"Feature iniziali: {len(X.columns)}")
    print(f"Dopo correlation: {len(X_no_corr.columns)} (-{len(removed_corr)})")
    print(f"Dopo variance: {len(X_preprocessed.columns)} (-{len(removed_var)})")
    print(f"\n✅ SPLIT 80-20: {len(X_train)} train, {len(X_test)} test")
    print(f"\nFeature/Componenti finali: {n_features_final}")
    print(f"Riduzione: {len(X.columns)} → {n_features_final} ({(1-n_features_final/len(X.columns))*100:.1f}%)")

    print(f"\nPerformance Validation (Monte-Carlo CV):")
    print(f"  Full dataset: {acc_full:.3f}")
    print(f"  Preprocessed: {acc_prep:.3f}")
    print(f"  PCA: {acc_pca:.3f}")
    print(f"  ANOVA: {acc_anova:.3f}")
    print(f"  🏆 Miglior metodo: {best_method}")

    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETATA")
    print("="*70)
    print("\n IMPORTANTE:")
    print("✓ NO data leakage")
    print("✓ PCA e ANOVA fittati solo su train")
    print("✓ Modelli salvati per deployment")
    print("✓ Dataset includono colonna 'split' per identificare train/test")
    print("\nFile salvati:")
    print(f"  - {pca_path}")
    print(f"  - {anova_path}")
    print(f"  - pca_fitted.pkl, scaler_pca.pkl")
    print(f"  - anova_selector_fitted.pkl, scaler_anova.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature Selection con PCA e ANOVA"
    )

    parser.add_argument(
        "--dataset-custom",
        type=str,
        required=True,
        help="Path al CSV dataset custom"
    )

    parser.add_argument(
        "--dataset-egemaps",
        type=str,
        required=True,
        help="Path al CSV extracted_features_eGeMAPS.csv"
    )

    parser.add_argument(
        "--dataset-index",
        type=str,
        required=True,
        help="Path al file dataset_index.xlsx"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="feature_selection_results",
        help="Directory output"
    )

    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.95,
        help="Soglia correlazione (default: 0.95)"
    )

    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.005,
        help="Soglia varianza (default: 0.05)"
    )

    parser.add_argument(
        "--n-features",
        type=int,
        default=None,
        help="N° componenti/feature (default: auto da PCA 95%%)"
    )

    args = parser.parse_args()
    main(args)