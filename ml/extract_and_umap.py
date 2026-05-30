import json
import os
import numpy as np
import pandas as pd
import librosa
from sklearn.decomposition       import PCA
from sklearn.manifold            import TSNE
from sklearn.cluster             import KMeans
from sklearn.mixture             import GaussianMixture
from sklearn.metrics             import silhouette_score, adjusted_rand_score
from sklearn.preprocessing       import StandardScaler
import umap

# ── Configuration ──────────────────────────────────────────────

VOCALSET_DIR   = "ml/VocalSet/FULL"
OUTPUT_JSON    = "data/zones.json"
N_MFCC         = 13
SR             = 22050
SEG_DUR        = 2.0
K_RANGE        = range(2, 21)

CLUSTER_COLORS = [
    "#7F77DD", "#1D9E75", "#E24B4A", "#BA7517", "#888780",
    "#378ADD", "#f59e0b", "#ec4899", "#10b981", "#6366f1",
    "#f97316", "#14b8a6", "#8b5cf6", "#ef4444", "#84cc16",
    "#0ea5e9", "#d946ef", "#fb923c", "#a3e635", "#38bdf8",
]

TECHNIQUES = ["breathy", "straight", "belt", "vibrato", "spoken"]

# ── Step 1: Extract MFCC → DataFrame ───────────────────────────

def extract_features(vocalset_dir):
    rows    = []
    singers = [s for s in os.listdir(vocalset_dir)
               if os.path.isdir(os.path.join(vocalset_dir, s))]
    print(f"Found {len(singers)} singer folders\n")

    for singer in sorted(singers):
        singer_path = os.path.join(vocalset_dir, singer)
        gender      = "female" if singer.startswith("female") else "male"

        for exercise in os.listdir(singer_path):
            exercise_path = os.path.join(singer_path, exercise)
            if not os.path.isdir(exercise_path):
                continue

            for technique in os.listdir(exercise_path):
                if technique not in TECHNIQUES:
                    continue
                tech_path = os.path.join(exercise_path, technique)

                for fname in os.listdir(tech_path):
                    if not fname.endswith('.wav'):
                        continue
                    try:
                        y, _ = librosa.load(
                            os.path.join(tech_path, fname), sr=SR, mono=True)
                        seg_len = int(SR * SEG_DUR)
                        for start in range(0, len(y) - seg_len, seg_len // 2):
                            seg   = y[start:start + seg_len]
                            mfcc  = librosa.feature.mfcc(
                                y=seg, sr=SR, n_mfcc=N_MFCC)
                            means = mfcc.mean(axis=1)
                            stds  = mfcc.std(axis=1)
                            row   = {
                                "singer":    singer,
                                "gender":    gender,
                                "technique": technique,
                            }
                            for i, v in enumerate(means):
                                row[f"mfcc_mean_{i}"] = v
                            for i, v in enumerate(stds):
                                row[f"mfcc_std_{i}"] = v
                            rows.append(row)
                    except Exception as e:
                        print(f"  Skipping {fname}: {e}")

    return pd.DataFrame(rows)

# ── Normalize embedding to [0.05, 0.95] ────────────────────────

def normalize_embedding(emb):
    result = emb.copy()
    for d in range(result.shape[1]):
        mn           = result[:, d].min()
        mx           = result[:, d].max()
        result[:, d] = (result[:, d] - mn) / (mx - mn) * 0.9 + 0.05
    return result

# ── Step 2: Best dimensionality reduction ──────────────────────

def best_reduction(X_scaled, technique_labels):
    methods = {
        "PCA":  PCA(n_components=2, random_state=42),
        "tSNE": TSNE(n_components=2, perplexity=30, random_state=42),
        "UMAP": umap.UMAP(n_components=2, n_neighbors=20,
                          min_dist=0.1, random_state=42),
    }
    best_name  = None
    best_score = -999
    best_emb   = None
    dr_results = {}

    print("\nStep 2: Comparing dimensionality reduction methods...")
    for name, reducer in methods.items():
        print(f"  Running {name}...")
        emb              = normalize_embedding(reducer.fit_transform(X_scaled))
        score            = silhouette_score(emb, technique_labels)
        dr_results[name] = round(float(score), 4)
        print(f"  Silhouette (predefined labels): {dr_results[name]}")
        if score > best_score:
            best_score = score
            best_name  = name
            best_emb   = emb

    print(f"\n  Best method: {best_name} ({best_score:.4f})")
    return best_emb, best_name, dr_results

# ── Step 3: KMeans vs GMM optimal K search ─────────────────────

def find_optimal_k(embedding, k_range):
    print(f"\nStep 3: KMeans vs GMM — K={k_range.start} to {k_range.stop - 1}...")
    print(f"  {'K':>3}  {'KMeans':>10}  {'GMM':>10}  {'Winner':>8}")
    print(f"  {'---':>3}  {'----------':>10}  {'----------':>10}  {'--------':>8}")

    km_scores  = {}
    gmm_scores = {}

    for k in k_range:
        km_labels       = KMeans(n_clusters=k, random_state=42,
                                 n_init=10).fit_predict(embedding)
        km_sil          = round(float(silhouette_score(embedding, km_labels)), 4)
        km_scores[k]    = km_sil

        gmm_labels      = GaussianMixture(n_components=k, random_state=42,
                                          n_init=3).fit_predict(embedding)
        gmm_sil         = round(float(silhouette_score(embedding, gmm_labels)), 4)
        gmm_scores[k]   = gmm_sil

        winner = "KMeans" if km_sil >= gmm_sil else "GMM   "
        print(f"  K={k:2d}  {km_sil:>10.4f}  {gmm_sil:>10.4f}  {winner}")

    best_km_k    = max(km_scores,  key=km_scores.get)
    best_gmm_k   = max(gmm_scores, key=gmm_scores.get)
    best_km_sil  = km_scores[best_km_k]
    best_gmm_sil = gmm_scores[best_gmm_k]

    print(f"\n  KMeans best: K={best_km_k}  score={best_km_sil}")
    print(f"  GMM    best: K={best_gmm_k}  score={best_gmm_sil}")

    if best_gmm_sil > best_km_sil:
        print(f"  → GMM wins (+{best_gmm_sil - best_km_sil:.4f})")
        print(f"    Clusters are elliptical — KMeans' spherical assumption was wrong")
        best_algo = "GMM"
        best_k    = best_gmm_k
    else:
        print(f"  → KMeans wins")
        print(f"    Spherical assumption holds — GMM adds no benefit")
        best_algo = "KMeans"
        best_k    = best_km_k

    return km_scores, gmm_scores, best_k, best_algo

# ── Analyze cluster composition ────────────────────────────────

def analyze_cluster(cluster_labels, group_labels, group_type, k):
    print(f"\n── Cluster composition by {group_type} ─────────")
    summary = {}
    for c in range(k):
        mask = cluster_labels == c
        if mask.sum() == 0:
            continue
        counts     = pd.Series(group_labels[mask]).value_counts()
        dominant   = counts.index[0]
        purity     = counts.iloc[0] / mask.sum() * 100
        summary[c] = {
            "dominant": dominant,
            "purity":   round(purity, 1),
            "n":        int(mask.sum()),
        }
        print(f"  Cluster {c:2d}: {dominant:12s} {purity:.0f}%  (n={mask.sum()})")
    return summary

# ── Per-singer normalization ────────────────────────────────────
# Subtracts each singer's mean from their own features.
# Removes individual voice identity so technique differences become clearer.
# NOTE: also removes mean pitch difference between male/female —
#       this is why gender ARI drops after normalization.

def normalize_per_singer(df, feat_cols):
    df_norm = df.copy()
    for singer in df["singer"].unique():
        mask               = df["singer"] == singer
        singer_mean        = df.loc[mask, feat_cols].mean()
        df_norm.loc[mask, feat_cols] = df.loc[mask, feat_cols] - singer_mean
    return df_norm

# ── Gender analysis ────────────────────────────────────────────

def gender_analysis(embedding, gender_labels, label):
    print(f"\n── Gender analysis on {label} ─────────────────")

    gender_binary = (gender_labels == "female").astype(int)
    results       = {}

    for algo_name, predictor in [
        ("KMeans K=2", KMeans(n_clusters=2, random_state=42, n_init=20)),
        ("GMM K=2",    GaussianMixture(n_components=2, random_state=42, n_init=5)),
    ]:
        pred = predictor.fit_predict(embedding)
        sil  = round(float(silhouette_score(embedding, pred)), 4)
        ari  = round(float(adjusted_rand_score(gender_binary, pred)), 4)

        for c in range(2):
            mask   = pred == c
            counts = pd.Series(gender_labels[mask]).value_counts()
            dom    = counts.index[0]
            pct    = counts.iloc[0] / mask.sum() * 100
            print(f"  [{algo_name}] Cluster {c}: "
                  f"{dom:8s} {pct:.0f}%  n={mask.sum()}")
        print(f"  [{algo_name}] Silhouette={sil}  ARI={ari}")

        key           = algo_name.replace(" ", "_").lower()
        results[key]  = {"silhouette": sil, "ari_vs_gender": ari}

    best_ari = max(v["ari_vs_gender"] for v in results.values())
    if   best_ari > 0.5:  verdict = "STRONG — separate male/female analysis recommended"
    elif best_ari > 0.2:  verdict = "MODERATE — some gender signal"
    elif best_ari > 0.05: verdict = "WEAK — loosely matches gender"
    else:                 verdict = "NONE — gender mixes freely in MFCC space"

    print(f"  Best ARI: {best_ari}  → {verdict}")
    results["best_ari"] = best_ari
    results["verdict"]  = verdict
    return results

# ── Build zones from cluster centers ───────────────────────────

def build_zones(embedding, cluster_labels, cluster_summary, k):
    zones = []
    for c in range(k):
        if c not in cluster_summary:
            continue
        mask   = cluster_labels == c
        cx     = float(embedding[mask, 0].mean())
        cy     = float(embedding[mask, 1].mean())
        spread = float(embedding[mask].std())
        radius = int(np.clip(spread * 150, 28, 65))
        info   = cluster_summary[c]
        zones.append({
            "name":     f"{info['dominant']} {c}",
            "x":        round(cx, 3),
            "y":        round(cy, 3),
            "color":    CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
            "radius":   radius,
            "dominant": info["dominant"],
            "purity":   info["purity"],
        })
    return zones

# ── Main ───────────────────────────────────────────────────────

def main():
    print("=== VOCALAB ML Pipeline ===\n")

    # 1. Extract features
    print("Step 1: Extracting MFCC features...")
    df        = extract_features(VOCALSET_DIR)
    feat_cols = [c for c in df.columns if c.startswith("mfcc")]
    print(f"  {len(df)} segments  |  {df['singer'].nunique()} singers")
    print(f"  Female: {(df['gender']=='female').sum()}  "
          f"Male: {(df['gender']=='male').sum()}")

    X                 = df[feat_cols].values
    technique_labels  = df["technique"].values
    singer_labels     = df["singer"].values
    gender_labels     = df["gender"].values
    X_scaled          = StandardScaler().fit_transform(X)

    # 2. Best dimensionality reduction
    embedding, best_dr, dr_results = best_reduction(X_scaled, technique_labels)

    # 3. KMeans vs GMM optimal K
    km_scores, gmm_scores, best_k, best_algo = find_optimal_k(embedding, K_RANGE)

    # 4. K=20 singer hypothesis
    print(f"\nStep 4: K=20 singer-clustering hypothesis...")
    labels_20  = KMeans(n_clusters=20, random_state=42,
                        n_init=10).fit_predict(embedding)
    singer_sum = analyze_cluster(labels_20, singer_labels, "singer", 20)
    avg_purity = np.mean([v["purity"] for v in singer_sum.values()])
    print(f"\n  K=20 singer purity: {avg_purity:.1f}%")
    print(f"  → {'CONFIRMED' if avg_purity > 60 else 'Not confirmed'}")

    # 5. Per-singer normalization
    print(f"\nStep 5: Per-singer normalization...")
    df_norm = normalize_per_singer(df, feat_cols)
    X_norm  = StandardScaler().fit_transform(df_norm[feat_cols].values)

    if   best_dr == "PCA":  reducer = PCA(n_components=2, random_state=42)
    elif best_dr == "tSNE": reducer = TSNE(n_components=2, perplexity=30,
                                           random_state=42)
    else:                   reducer = umap.UMAP(n_components=2, n_neighbors=20,
                                                min_dist=0.1, random_state=42)

    emb_norm   = normalize_embedding(reducer.fit_transform(X_norm))
    sil_before = dr_results[best_dr]
    sil_after  = round(float(silhouette_score(emb_norm, technique_labels)), 4)
    print(f"  Silhouette BEFORE normalization: {sil_before}")
    print(f"  Silhouette AFTER  normalization: {sil_after}")

    use_emb = emb_norm if sil_after > sil_before else embedding
    note    = "per-singer normalized" if sil_after > sil_before else "raw"

    # 6. Gender analysis — RAW embedding
    print("\nStep 6: Gender analysis...")
    gender_raw  = gender_analysis(embedding, gender_labels, "RAW embedding")

    # 7. Gender analysis — NORMALIZED embedding
    gender_norm = gender_analysis(emb_norm,  gender_labels, "NORMALIZED embedding")

    # Compare the two
    print(f"\n── Effect of normalization on gender signal ──")
    print(f"  ARI before normalization: {gender_raw['best_ari']}")
    print(f"  ARI after  normalization: {gender_norm['best_ari']}")
    ari_drop = gender_raw['best_ari'] - gender_norm['best_ari']
    if ari_drop > 0.05:
        print(f"  → Normalization REMOVED gender signal (drop={ari_drop:.4f})")
        print(f"    Mean pitch difference between male/female was a real signal")
    else:
        print(f"  → Gender signal was already absent before normalization")
        print(f"    MFCC does not capture male/female differences")

    # 8. Final clustering
    print(f"\nStep 8: Final clustering — {best_algo} K={best_k} on {note}...")
    if best_algo == "GMM":
        final_labels = GaussianMixture(n_components=best_k, random_state=42,
                                       n_init=5).fit_predict(use_emb)
    else:
        final_labels = KMeans(n_clusters=best_k, random_state=42,
                              n_init=20).fit_predict(use_emb)

    final_sil = round(float(silhouette_score(use_emb, final_labels)), 4)
    print(f"  Final Silhouette: {final_sil}")
    final_sum = analyze_cluster(
        final_labels, technique_labels, "technique", best_k)
    zones = build_zones(use_emb, final_labels, final_sum, best_k)

    # 9. Write zones.json
    output = {
        "_comment":               "Generated by extract_and_umap.py",
        "_dr_results":            dr_results,
        "_km_scores":             {str(k): v for k, v in km_scores.items()},
        "_gmm_scores":            {str(k): v for k, v in gmm_scores.items()},
        "_optimal_k":             best_k,
        "_best_algo":             best_algo,
        "_k20_singer_purity":     round(avg_purity, 1),
        "_normalization":         note,
        "_final_silhouette":      final_sil,
        "_gender_raw":            gender_raw,
        "_gender_normalized":     gender_norm,
        "_gender_ari_drop":       round(ari_drop, 4),
        "best_dr_method":         best_dr,
        "zones":                  zones,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Written to {OUTPUT_JSON}")

    print(f"\n═══ Summary ════════════════════════════════════════")
    print(f"  Best DR                 : {best_dr}")
    print(f"  Predefined Silhouette   : {sil_before}")
    print(f"  Best KMeans K           : {max(km_scores,  key=km_scores.get)}"
          f"  score={max(km_scores.values())}")
    print(f"  Best GMM K              : {max(gmm_scores, key=gmm_scores.get)}"
          f"  score={max(gmm_scores.values())}")
    print(f"  Winner algorithm        : {best_algo}")
    print(f"  K=20 singer purity      : {avg_purity:.1f}%")
    print(f"  Silhouette after norm   : {sil_after}")
    print(f"  Gender ARI (raw)        : {gender_raw['best_ari']}")
    print(f"  Gender ARI (normalized) : {gender_norm['best_ari']}")
    print(f"  Gender ARI drop         : {ari_drop:.4f}")
    print(f"  Final Silhouette        : {final_sil}")

if __name__ == "__main__":
    main()