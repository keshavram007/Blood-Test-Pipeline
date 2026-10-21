import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt

NUMERIC_FEATURES = [
    'Age', 'Triglycerides', 'TotalCholesterol', 'UricAcid', 'Albumin',
    'Creatinine', 'Alanine Aminotransferase', 'HDLCholesterol',
    'Sodium', 'BUN', 'GGT', 'AST'
]

NUMERIC_FEATURES_WITH_GLUCOSE = NUMERIC_FEATURES + ['FastingGlucose', 'Glycated Haemoglobin']


def load_and_prepare():
    df = pd.read_csv('data/processed/labeled_dataset.csv')
    df['Gender_encoded'] = df['Gender'].map({'Male': 0, 'Female': 1})
    feature_cols = NUMERIC_FEATURES + ['Gender_encoded']
    X = StandardScaler().fit_transform(df[feature_cols])
    return df, X


def find_optimal_k(X, k_range=range(2, 8)):
    """Elbow method — does the data naturally want 3 clusters, or a different number?"""
    inertias = []
    silhouette_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil = silhouette_score(X, labels, sample_size=5000, random_state=42)
        silhouette_scores.append(sil)
        print(f"k={k}: inertia={km.inertia_:.1f}, silhouette={sil:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(list(k_range), inertias, marker='o')
    axes[0].set_xlabel('Number of clusters (k)')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Method')

    axes[1].plot(list(k_range), silhouette_scores, marker='o', color='orange')
    axes[1].set_xlabel('Number of clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Analysis')

    plt.tight_layout()
    plt.savefig('experiments/output/figures/clustering_k_selection.png', dpi=150)
    print("\nSaved: experiments/output/figures/clustering_k_selection.png")

    return inertias, silhouette_scores


def run_kmeans_k3(df, X):
    """Run K-Means with k=3 and compare against actual DiabetesStatus labels."""
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)

    print("\nCluster vs. actual DiabetesStatus crosstab:")
    crosstab = pd.crosstab(df['Cluster'], df['DiabetesStatus'])
    print(crosstab)

    print("\nCluster vs. actual DiabetesStatus (row-normalized %):")
    print((crosstab.div(crosstab.sum(axis=1), axis=0) * 100).round(1))

    ari = adjusted_rand_score(df['DiabetesStatus'], df['Cluster'])
    print(f"\nAdjusted Rand Index (cluster vs. true label agreement): {ari:.4f}")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='viridis', alpha=0.3, s=5)
    axes[0].set_title('K-Means Clusters (k=3)')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    plt.colorbar(scatter1, ax=axes[0])

    label_map = {'Normal': 0, 'Prediabetic': 1, 'Diabetic': 2}
    colors = df['DiabetesStatus'].map(label_map)
    scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=colors, cmap='viridis', alpha=0.3, s=5)
    axes[1].set_title('Actual DiabetesStatus Labels')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    plt.colorbar(scatter2, ax=axes[1])

    plt.tight_layout()
    plt.savefig('experiments/output/figures/clustering_vs_labels.png', dpi=150)
    print("Saved: experiments/output/figures/clustering_vs_labels.png")

    return df, kmeans, ari


def profile_clusters(df, cluster_col='Cluster', features=NUMERIC_FEATURES):
    """Show mean biomarker values per cluster — helps interpret what each cluster represents."""
    print(f"\nCluster profiles (mean values) [{cluster_col}]:")
    profile = df.groupby(cluster_col)[features].mean().round(2)
    print(profile)
    return profile


def run_clustering_with_glucose():
    """Comparison run: same clustering approach, but WITH Glucose/HbA1c included."""
    df = pd.read_csv('data/processed/labeled_dataset.csv')
    df['Gender_encoded'] = df['Gender'].map({'Male': 0, 'Female': 1})
    feature_cols = NUMERIC_FEATURES_WITH_GLUCOSE + ['Gender_encoded']

    X = StandardScaler().fit_transform(df[feature_cols])

    print("\n--- Optimal k check (WITH glucose/HbA1c) ---")
    for k in range(2, 6):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels, sample_size=5000, random_state=42)
        print(f"k={k}: inertia={km.inertia_:.1f}, silhouette={sil:.4f}")

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster_withglucose'] = kmeans.fit_predict(X)

    print("\nCluster vs. actual DiabetesStatus crosstab (WITH glucose/HbA1c):")
    crosstab = pd.crosstab(df['Cluster_withglucose'], df['DiabetesStatus'])
    print(crosstab)

    print("\nCluster vs. actual DiabetesStatus (row-normalized %):")
    print((crosstab.div(crosstab.sum(axis=1), axis=0) * 100).round(1))

    ari = adjusted_rand_score(df['DiabetesStatus'], df['Cluster_withglucose'])
    print(f"\nAdjusted Rand Index (WITH glucose/HbA1c): {ari:.4f}")

    profile_clusters(df, cluster_col='Cluster_withglucose', features=NUMERIC_FEATURES_WITH_GLUCOSE)

    return df, kmeans, ari


if __name__ == '__main__':
    print("=" * 60)
    print("PART A: CLUSTERING WITHOUT GLUCOSE/HbA1c (conservative feature set)")
    print("=" * 60)

    df, X = load_and_prepare()

    print("\nSTEP 1: Finding optimal k")
    find_optimal_k(X)

    print("\nSTEP 2: K-Means with k=3, compared to actual labels")
    df, kmeans, ari_no_glucose = run_kmeans_k3(df, X)

    print("\nSTEP 3: Cluster profiles")
    profile_clusters(df)

    print("\n\n" + "=" * 60)
    print("PART B: CLUSTERING WITH GLUCOSE/HbA1c INCLUDED (comparison)")
    print("=" * 60)

    df_glucose, kmeans_glucose, ari_with_glucose = run_clustering_with_glucose()

    print("\n\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"ARI WITHOUT Glucose/HbA1c: {ari_no_glucose:.4f}")
    print(f"ARI WITH Glucose/HbA1c:    {ari_with_glucose:.4f}")