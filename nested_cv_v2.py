"""
EEG-fNIRS AD Classification — v2: Nested + Repeated Cross-Validation
=====================================================================
Addresses reviewer comments:
  #1  Model-selection bias  -> k and all hyperparameters chosen in an INNER loop;
                                the OUTER test fold never touches selection.
  #2  Hyperparameter tuning -> explicit grids for LR / LASSO / RF / SVM, tuned per fold.
  #3  Uncertainty           -> 5x5 repeated stratified CV, per-fold metrics,
                                mean, SD, 95% CI.
  #4  Statistical tests     -> Wilcoxon signed-rank + Nadeau-Bengio corrected t-test
                                on identical outer folds.
  #6  Modality evidence     -> permutation importance, cross-fold stability (Jaccard),
                                modality-group permutation (EEG vs fNIRS block).
  #7  Reproducibility       -> all seeds fixed (incl. MI), library versions logged.

Design notes
------------
* Metrics are computed PER OUTER FOLD (not pooled), so that CIs and paired tests are possible.
* AUC: macro one-vs-rest. predict_proba for LR/LASSO/RF; decision_function for SVM
  (Platt scaling not used -> avoids 5-fold internal CV inside SVC).
* LASSO = one-vs-rest L1 logistic regression (liblinear). The original saga/multinomial
  variant needed 10-15 s per fit on 4,968 features and was infeasible in a nested loop.
* Untuned "default" pipelines (k=50, original hyperparameters) are evaluated on the SAME
  outer folds for a tuned-vs-untuned comparison.
"""

import os, sys, time, json, platform, warnings
from collections import defaultdict
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import sklearn
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
QUICK = os.environ.get("QUICK", "0") == "1"          # smoke-test mode

DATA_PATH = os.environ.get("DATA_PATH", "./5secEEGPSD_FullFnirsPSD_FullFnirsTimeDomain.csv")
OUT_DIR     = os.environ.get("OUT_DIR", "./results_v2" + ("_quick" if QUICK else ""))
SEED        = 42
N_OUTER     = 5
N_REPEATS   = 1 if QUICK else 5
N_INNER     = 3
LABEL_REMAP = {0: 2, 1: 0, 2: 1}   # raw 0=AD,1=HC,2=MCI  ->  HC=0, MCI=1, AD=2
CLASS_NAMES_3 = ["HC", "MCI", "AD"]
CLASS_NAMES_2 = ["HC", "MCI"]

K_GRID       = [20, 100] if QUICK else [10, 20, 50, 100, 200, 500]       # inner-loop k grid
K_SENS       = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500]              # sensitivity curve (Fig.1)
FS_TYPES     = ["none", "anova", "mi"]
CLF_NAMES    = ["LR", "LASSO", "RF", "SVM"]

if QUICK:
    GRIDS = {
        "LR":    [{"C": c} for c in [0.1, 1]],
        "LASSO": [{"C": c} for c in [0.5, 1]],
        "RF":    [{"max_depth": d, "min_samples_leaf": l} for d in [10] for l in [1, 5]],
        "SVM":   [{"C": c, "gamma": g} for c in [1, 10] for g in ["scale"]],
    }
else:
    GRIDS = {
        "LR":    [{"C": c} for c in [0.01, 0.1, 1, 10]],
        "LASSO": [{"C": c} for c in [0.1, 0.5, 1, 5]],
        "RF":    [{"max_depth": d, "min_samples_leaf": l} for d in [5, 10, None] for l in [1, 5]],
        "SVM":   [{"C": c, "gamma": g} for c in [0.1, 1, 10, 100] for g in ["scale", 1e-4, 1e-3, 1e-2]],
    }
# "default" = the untuned configuration used in the original manuscript
DEFAULTS = {
    "LR":    {"C": 1.0},
    "LASSO": {"C": 1.0},
    "RF":    {"max_depth": 10, "min_samples_leaf": 5},
    "SVM":   {"C": 1.0, "gamma": "scale"},
}
DEFAULT_K = 50
N_TREES   = 200
PERM_REPEATS_FEATURE  = 1 if QUICK else 3     # per-feature permutation importance repeats
PERM_FOLDS_FEATURE    = 1 if QUICK else 5     # only repeat-1 folds (cost: 4968 feats x repeats x folds)
PERM_REPEATS_MODALITY = 2 if QUICK else 10    # modality-block permutation repeats (all folds)
TOP_N = 20

os.makedirs(OUT_DIR, exist_ok=True)
LOG = open(os.path.join(OUT_DIR, "run_log.txt"), "a")

def log(msg=""):
    print(msg); LOG.write(msg + "\n"); LOG.flush(); sys.stdout.flush()

# =============================================================================
# DATA
# =============================================================================
def load_data():
    df = pd.read_csv(DATA_PATH)
    drop = ["label"] + [c for c in df.columns if c.startswith("Unnamed")]
    X = df.drop(columns=drop).values.astype(float)
    names = df.drop(columns=drop).columns.tolist()
    y = np.array([LABEL_REMAP[v] for v in df["label"].values])
    n_nan, n_inf = int(np.isnan(X).sum()), int(np.isinf(X).sum())
    log(f"Data: X={X.shape}, HC={np.sum(y==0)}, MCI={np.sum(y==1)}, AD={np.sum(y==2)}")
    log(f"EEG feats={sum(n.startswith('eeg') for n in names)}, fNIRS feats={sum(n.startswith('fnirs') for n in names)}")
    log(f"NaN={n_nan}, Inf={n_inf}  (dropped columns: {drop})")
    assert n_nan == 0 and n_inf == 0
    return X, y, names

# =============================================================================
# MODELS & SCORING
# =============================================================================
def make_clf(name, p):
    if name == "LR":
        return LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000, C=p["C"])
    if name == "LASSO":
        return OneVsRestClassifier(LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000, C=p["C"]))
    if name == "RF":
        return RandomForestClassifier(n_estimators=N_TREES, max_depth=p["max_depth"],
                                      min_samples_leaf=p["min_samples_leaf"], random_state=SEED, n_jobs=1)
    if name == "SVM":
        return SVC(kernel="rbf", C=p["C"], gamma=p["gamma"], decision_function_shape="ovr")
    raise ValueError(name)

def scores_of(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return model.decision_function(X)

def macro_auc(y, S):
    classes = np.unique(y)
    if len(classes) == 2:
        s = S if S.ndim == 1 else S[:, 1]
        return roc_auc_score(y, s)
    return float(np.mean([roc_auc_score((y == c).astype(int), S[:, c]) for c in classes]))

def f1_of(y, yp):
    return f1_score(y, yp, average="binary" if len(np.unique(y)) == 2 else "macro")

def rank_features(fs, X, y):
    if fs == "anova":
        F, _ = f_classif(X, y); F = np.nan_to_num(F, nan=0.0)
        return np.argsort(-F)
    if fs == "mi":
        mi = mutual_info_classif(X, y, random_state=SEED)
        return np.argsort(-mi)
    return None

def select_cols(rank, k, n_feat):
    return np.arange(n_feat) if rank is None else rank[:k]

# =============================================================================
# NESTED CV CORE
# =============================================================================
def nested_cv(X, y, outer_splits, fs_types, clf_names, tag):
    """
    For each outer fold: inner 3-fold CV selects (k, hyperparams) jointly per (fs, clf)
    by mean inner macro-AUC; refit on full outer-train; evaluate on outer-test.
    Also evaluates the untuned DEFAULT pipeline on the same folds.
    Resumable: completed folds are loaded from per_fold_{tag}.csv and skipped.
    """
    path = os.path.join(OUT_DIR, f"per_fold_{tag}.csv")
    results, done = [], set()
    if os.path.exists(path):
        prev = pd.read_csv(path)
        n_expected = len(fs_types) * len(clf_names) * 2
        cnt = prev.groupby("fold_id").size()
        done = set(cnt[cnt == n_expected].index.tolist())
        results = prev[prev.fold_id.isin(done)].to_dict("records")
        log(f"[{tag}] resume: {len(done)}/{len(outer_splits)} folds already complete")
    n_feat = X.shape[1]
    t_start = time.time(); n_done_now = 0
    for fold_id, (tr, te) in enumerate(outer_splits):
        if fold_id in done:
            continue
        t0 = time.time()
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        rep, fold = fold_id // N_OUTER, fold_id % N_OUTER

        # ---------------- inner loop ----------------
        inner = StratifiedKFold(N_INNER, shuffle=True, random_state=SEED + fold_id)
        inner_auc = defaultdict(list)
        for itr, ival in inner.split(Xtr, ytr):
            sc = StandardScaler().fit(Xtr[itr])
            A_full, B_full = sc.transform(Xtr[itr]), sc.transform(Xtr[ival])
            ranks = {fs: rank_features(fs, A_full, ytr[itr]) for fs in fs_types}
            for fs in fs_types:
                for k in (K_GRID if fs != "none" else [None]):
                    cols = select_cols(ranks[fs], k, n_feat)
                    A, B = A_full[:, cols], B_full[:, cols]
                    for clf in clf_names:
                        for pi, p in enumerate(GRIDS[clf]):
                            m = make_clf(clf, p).fit(A, ytr[itr])
                            inner_auc[(fs, k, clf, pi)].append(macro_auc(ytr[ival], scores_of(m, B)))

        # ---------------- outer refit ----------------
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        ranks = {fs: rank_features(fs, Xtr_s, ytr) for fs in fs_types}

        for fs in fs_types:
            for clf in clf_names:
                # --- tuned ---
                cands = [(np.mean(v), key) for key, v in inner_auc.items() if key[0] == fs and key[2] == clf]
                best_auc_inner, (_, k_best, _, pi_best) = max(cands, key=lambda t: t[0])
                p_best = GRIDS[clf][pi_best]
                cols = select_cols(ranks[fs], k_best, n_feat)
                m = make_clf(clf, p_best).fit(Xtr_s[:, cols], ytr)
                S, yp = scores_of(m, Xte_s[:, cols]), m.predict(Xte_s[:, cols])
                results.append(dict(task=tag, fold_id=fold_id, repeat=rep, fold=fold, fs=fs, clf=clf, tuned=True,
                                    k=k_best if fs != "none" else n_feat, params=json.dumps(p_best),
                                    inner_auc=best_auc_inner,
                                    acc=accuracy_score(yte, yp), f1=f1_of(yte, yp), auc=macro_auc(yte, S),
                                    cm=json.dumps(confusion_matrix(yte, yp, labels=np.unique(y)).tolist())))
                # --- untuned default (original manuscript configuration) ---
                k_def = DEFAULT_K if fs != "none" else None
                cols = select_cols(ranks[fs], k_def, n_feat)
                m = make_clf(clf, DEFAULTS[clf]).fit(Xtr_s[:, cols], ytr)
                S, yp = scores_of(m, Xte_s[:, cols]), m.predict(Xte_s[:, cols])
                results.append(dict(task=tag, fold_id=fold_id, repeat=rep, fold=fold, fs=fs, clf=clf, tuned=False,
                                    k=k_def if fs != "none" else n_feat, params=json.dumps(DEFAULTS[clf]),
                                    inner_auc=np.nan,
                                    acc=accuracy_score(yte, yp), f1=f1_of(yte, yp), auc=macro_auc(yte, S),
                                    cm=json.dumps(confusion_matrix(yte, yp, labels=np.unique(y)).tolist())))
        pd.DataFrame(results).to_csv(path, index=False)
        n_done_now += 1; el = time.time() - t_start; remaining = len(outer_splits) - fold_id - 1
        log(f"[{tag}] fold {fold_id+1}/{len(outer_splits)} done in {time.time()-t0:.0f}s "
            f"(elapsed {el/60:.1f} min, ETA {el/n_done_now*remaining/60:.1f} min)")
    return results

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================
def ci95(v):
    v = np.asarray(v, float); n = len(v)
    if n < 2: return (np.nan, np.nan)
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return (v.mean() - h, v.mean() + h)

def summarize(df, group_cols):
    rows = []
    for key, g in df.groupby(group_cols, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        r = dict(zip(group_cols, key))
        for met in ["acc", "f1", "auc"]:
            lo, hi = ci95(g[met])
            r[f"{met}_mean"] = g[met].mean(); r[f"{met}_sd"] = g[met].std(ddof=1)
            r[f"{met}_ci_lo"] = lo; r[f"{met}_ci_hi"] = hi
        r["n_folds"] = len(g)
        if "k" in g: r["k_median"] = g["k"].median(); r["k_iqr"] = f"{g['k'].quantile(.25):.0f}-{g['k'].quantile(.75):.0f}"
        if "params" in g: r["params_mode"] = g["params"].mode().iloc[0]; r["params_mode_freq"] = (g["params"] == r["params_mode"]).mean()
        rows.append(r)
    return pd.DataFrame(rows)

def paired_tests(a, b, n_train, n_test):
    """a, b: per-fold metric arrays on identical folds."""
    d = np.asarray(a) - np.asarray(b); J = len(d)
    try:
        w_stat, w_p = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    # Nadeau & Bengio (2003) corrected resampled t-test
    var = d.var(ddof=1)
    denom = np.sqrt((1.0 / J + n_test / n_train) * var) if var > 0 else np.nan
    t_nb = d.mean() / denom if denom and denom > 0 else np.nan
    p_nb = 2 * stats.t.sf(abs(t_nb), J - 1) if np.isfinite(t_nb) else np.nan
    return dict(mean_diff=d.mean(), sd_diff=d.std(ddof=1), ci_lo=ci95(d)[0], ci_hi=ci95(d)[1],
                wilcoxon_p=w_p, nb_t=t_nb, nb_p=p_nb, n_folds=J)

def holm(pvals):
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); running = 0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * p[idx]); adj[idx] = min(1.0, running)
    return adj

# =============================================================================
# EXPERIMENTS
# =============================================================================
def exp_sensitivity_curve(X, y, outer_splits):
    """Fig.1 replacement: ANOVA(k fixed) + LR(C=1) per k, on the outer folds only.
    No selection is performed from this curve -> descriptive, unbiased per-k estimates."""
    log("\n=== Sensitivity curve: ANOVA(k) + LR, repeated CV ===")
    sens_path = os.path.join(OUT_DIR, "sens_per_fold.csv")
    rows = []
    if os.path.exists(sens_path) and pd.read_csv(sens_path)["fold_id"].nunique() == len(outer_splits):
        rows = pd.read_csv(sens_path).to_dict("records"); log("  (loaded from checkpoint)")
    for fold_id, (tr, te) in (enumerate(outer_splits) if not rows else []):
        sc = StandardScaler().fit(X[tr]); A, B = sc.transform(X[tr]), sc.transform(X[te])
        rank = rank_features("anova", A, y[tr])
        for k in K_SENS:
            cols = rank[:k]
            m = make_clf("LR", {"C": 1.0}).fit(A[:, cols], y[tr])
            yp = m.predict(B[:, cols])
            rows.append(dict(fold_id=fold_id, k=k, acc=accuracy_score(y[te], yp), f1=f1_of(y[te], yp),
                             auc=macro_auc(y[te], scores_of(m, B[:, cols]))))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT_DIR, "sens_per_fold.csv"), index=False)
    summ = summarize(df, ["k"]); summ.to_csv(os.path.join(OUT_DIR, "table1_sensitivity_k.csv"), index=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for met, c, lab in [("acc", "#4C72B0", "Accuracy"), ("f1", "#DD8452", "Macro F1"), ("auc", "#55A868", "Macro AUC")]:
        ax.plot(summ["k"], summ[f"{met}_mean"], "o-", color=c, label=lab)
        ax.fill_between(summ["k"], summ[f"{met}_ci_lo"], summ[f"{met}_ci_hi"], color=c, alpha=0.15)
    ax.set_xscale("log"); ax.set_xticks(K_SENS); ax.set_xticklabels(K_SENS)
    ax.set_xlabel("Number of ANOVA-selected features k (log scale)"); ax.set_ylabel("Score")
    ax.set_title(f"Three-class ANOVA + LR sensitivity to k\n(mean and 95% CI over {N_OUTER}x{N_REPEATS} repeated stratified CV)")
    ax.grid(alpha=.3); ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig1_sensitivity_k.png"), dpi=200); plt.close()
    log(summ[["k", "auc_mean", "auc_sd", "auc_ci_lo", "auc_ci_hi"]].round(4).to_string(index=False))
    return summ

def plot_cm(df_task, fs, clf, class_names, path, title):
    g = df_task[(df_task.fs == fs) & (df_task.clf == clf) & (df_task.tuned)]
    cm = np.sum([np.array(json.loads(c)) for c in g["cm"]], axis=0) / N_REPEATS   # -> counts on 144 (or 109) scale
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]:.1f}\n({cm_norm[i,j]*100:.1f}%)", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names)
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046); plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
    return cm, cm_norm

def exp_modality(X, y, names, outer_splits):
    """Table IV / Fig.4: EEG-only vs fNIRS-only vs multimodal, nested, two pipelines."""
    log("\n=== Modality comparison (nested): ANOVA+LR and None+RF ===")
    eeg = [i for i, n in enumerate(names) if n.startswith("eeg")]
    fnirs = [i for i, n in enumerate(names) if n.startswith("fnirs")]
    all_res = []
    for mod, idx in [("EEG-only", eeg), ("fNIRS-only", fnirs), ("Multimodal", eeg + fnirs)]:
        Xm = X[:, idx]
        r1 = nested_cv(Xm, y, outer_splits, ["anova"], ["LR"], f"mod_{mod}_anova_LR")
        r2 = nested_cv(Xm, y, outer_splits, ["none"], ["RF"], f"mod_{mod}_none_RF")
        for r in r1 + r2: r["modality"] = mod; r["n_features"] = len(idx)
        all_res += r1 + r2
    df = pd.DataFrame(all_res); df.to_csv(os.path.join(OUT_DIR, "per_fold_modality.csv"), index=False)
    summ = summarize(df[df.tuned], ["modality", "fs", "clf"]); summ.to_csv(os.path.join(OUT_DIR, "table4_modality.csv"), index=False)
    log(summ[["modality", "fs", "clf", "auc_mean", "auc_sd", "auc_ci_lo", "auc_ci_hi", "k_median"]].round(4).to_string(index=False))
    # figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (fs, clf, ttl) in zip(axes, [("anova", "LR", "ANOVA + LR (k, C tuned)"), ("none", "RF", "No selection + RF (tuned)")]):
        s = summ[(summ.fs == fs) & (summ.clf == clf)].set_index("modality").loc[["EEG-only", "fNIRS-only", "Multimodal"]]
        x = np.arange(3); w = .26
        for j, (met, c, lab) in enumerate([("acc", "#4C72B0", "Accuracy"), ("f1", "#DD8452", "Macro F1"), ("auc", "#55A868", "Macro AUC")]):
            err = [s[f"{met}_mean"] - s[f"{met}_ci_lo"], s[f"{met}_ci_hi"] - s[f"{met}_mean"]]
            ax.bar(x + (j - 1) * w, s[f"{met}_mean"], w, yerr=err, capsize=3, color=c, label=lab)
        ax.set_xticks(x); ax.set_xticklabels(s.index); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=.3); ax.set_title(ttl, fontsize=10)
    axes[0].set_ylabel("Score (mean, 95% CI)"); axes[0].legend(loc="upper left", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig4_modality.png"), dpi=200); plt.close()
    return df, summ

def jaccard_sets(sets):
    if len(sets) < 2: return np.nan
    return float(np.mean([len(a & b) / len(a | b) for a, b in combinations(sets, 2)]))

def exp_importance(X, y, names, outer_splits, df3):
    """Reviewer #6: stability + permutation importance for the tuned No-selection RF (three-class).
    The evaluated RF of each outer fold is rebuilt deterministically from its recorded params (seed fixed).
    Checkpointed per fold in importance_per_fold.jsonl."""
    log("\n=== Feature-importance stability & permutation importance ===")
    n_feat = len(names)
    is_eeg = np.array([n.startswith("eeg") for n in names])
    ckpt = os.path.join(OUT_DIR, "importance_per_fold.jsonl")
    recs = {}
    if os.path.exists(ckpt):
        for line in open(ckpt):
            r = json.loads(line); recs[r["fold_id"]] = r
        log(f"  resume: {len(recs)} folds already done")
    t0 = time.time()
    for fold_id, (tr, te) in enumerate(outer_splits):
        if fold_id in recs:
            continue
        rng = np.random.default_rng(SEED + fold_id)
        row = df3[(df3.fold_id == fold_id) & (df3.fs == "none") & (df3.clf == "RF") & (df3.tuned == True)].iloc[0]
        p = json.loads(row["params"])
        sc = StandardScaler().fit(X[tr]); Xtr_s, Xte_s, ytr, yte = sc.transform(X[tr]), sc.transform(X[te]), y[tr], y[te]
        m = make_clf("RF", p).fit(Xtr_s, ytr)
        base = macro_auc(yte, m.predict_proba(Xte_s))
        assert abs(base - row["auc"]) < 1e-9, f"rebuilt model differs from evaluated one (fold {fold_id}): {base} vs {row['auc']}"
        anova_rank = rank_features("anova", Xtr_s, ytr)
        rec = dict(fold_id=fold_id, repeat=fold_id // N_OUTER, base_auc=base,
                   top_anova=[int(i) for i in anova_rank[:TOP_N]],
                   top_impurity=[int(i) for i in np.argsort(-m.feature_importances_)[:TOP_N]],
                   top_perm=None, block={})
        for blk, mask in [("EEG", is_eeg), ("fNIRS", ~is_eeg)]:
            drops = []
            for _ in range(PERM_REPEATS_MODALITY):
                Xp = Xte_s.copy(); Xp[:, mask] = Xp[rng.permutation(len(yte))][:, mask]
                drops.append(base - macro_auc(yte, m.predict_proba(Xp)))
            rec["block"][blk] = float(np.mean(drops))
        if rec["repeat"] == 0 and fold_id < PERM_FOLDS_FEATURE:
            imp = np.zeros(n_feat)
            for j in range(n_feat):
                d = 0.0
                for _ in range(PERM_REPEATS_FEATURE):
                    Xp = Xte_s.copy(); Xp[:, j] = Xp[rng.permutation(len(yte)), j]
                    d += base - macro_auc(yte, m.predict_proba(Xp))
                imp[j] = d / PERM_REPEATS_FEATURE
            rec["top_perm"] = [int(i) for i in np.argsort(-imp)[:TOP_N]]
        recs[fold_id] = rec
        with open(ckpt, "a") as f: f.write(json.dumps(rec) + "\n")
        log(f"  importance fold {fold_id+1}/{len(outer_splits)} done ({time.time()-t0:.0f}s)"
            + (f"  [per-feature perm: EEG in top-{TOP_N} = {is_eeg[rec['top_perm']].sum()}]" if rec["top_perm"] else ""))

    # ---------------- aggregate ----------------
    recs = [recs[i] for i in sorted(recs)]
    top_sets = {"ANOVA-F": [set(r["top_anova"]) for r in recs],
                "RF-impurity": [set(r["top_impurity"]) for r in recs],
                "RF-permutation": [set(r["top_perm"]) for r in recs if r["top_perm"]]}
    rows_stab = [dict(fold_id=r["fold_id"], method=meth, eeg_in_top=int(is_eeg[list(st)].sum()))
                 for meth, key in [("ANOVA-F", "top_anova"), ("RF-impurity", "top_impurity"), ("RF-permutation", "top_perm")]
                 for r in recs if r[key] for st in [set(r[key])]]
    df_stab = pd.DataFrame(rows_stab); df_stab.to_csv(os.path.join(OUT_DIR, "importance_modality_per_fold.csv"), index=False)
    df_blk = pd.DataFrame([dict(fold_id=r["fold_id"], block=b, base_auc=r["base_auc"], delta_auc=d) for r in recs for b, d in r["block"].items()])
    df_blk.to_csv(os.path.join(OUT_DIR, "importance_block_permutation_per_fold.csv"), index=False)
    df_perm = pd.DataFrame([dict(fold_id=r["fold_id"],
                                 jaccard_perm_vs_impurity=len(set(r["top_perm"]) & set(r["top_impurity"])) / len(set(r["top_perm"]) | set(r["top_impurity"])),
                                 jaccard_perm_vs_anova=len(set(r["top_perm"]) & set(r["top_anova"])) / len(set(r["top_perm"]) | set(r["top_anova"])))
                            for r in recs if r["top_perm"]])
    df_perm.to_csv(os.path.join(OUT_DIR, "importance_perm_vs_others.csv"), index=False)
    freq = {m: np.zeros(n_feat) for m in top_sets}
    for m, sets in top_sets.items():
        for st in sets: freq[m][list(st)] += 1
    agg = []
    for meth, sets in top_sets.items():
        eeg_counts = df_stab[df_stab.method == meth]["eeg_in_top"].values
        agg.append(dict(method=meth, n_folds=len(sets), mean_jaccard_across_folds=jaccard_sets(sets),
                        eeg_in_top_mean=eeg_counts.mean(), eeg_in_top_sd=eeg_counts.std(ddof=1) if len(eeg_counts) > 1 else 0,
                        fnirs_in_top_mean=TOP_N - eeg_counts.mean()))
    df_agg = pd.DataFrame(agg); df_agg.to_csv(os.path.join(OUT_DIR, "importance_stability_summary.csv"), index=False)
    consensus = {m: set(np.argsort(-freq[m])[:TOP_N]) for m in freq}
    for m in freq:
        idx = np.argsort(-freq[m])[:TOP_N]
        pd.DataFrame({"feature": [names[i] for i in idx], "freq_in_topN": freq[m][idx], "n_folds": len(top_sets[m])}).to_csv(
            os.path.join(OUT_DIR, f"top{TOP_N}_consensus_{m}.csv"), index=False)
    ov = pd.DataFrame([dict(a=a, b=b, overlap=len(consensus[a] & consensus[b])) for a, b in combinations(consensus, 2)])
    ov.to_csv(os.path.join(OUT_DIR, "importance_consensus_overlap.csv"), index=False)
    blk_summ = df_blk.groupby("block")["delta_auc"].agg(["mean", "std"]).reset_index()
    blk_summ["ci_lo"], blk_summ["ci_hi"] = zip(*[ci95(df_blk[df_blk.block == b]["delta_auc"]) for b in blk_summ["block"]])
    blk_summ.to_csv(os.path.join(OUT_DIR, "importance_block_permutation_summary.csv"), index=False)
    log(df_agg.round(3).to_string(index=False)); log(ov.to_string(index=False)); log(blk_summ.round(4).to_string(index=False))
    # ---- Fig.5 ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]; x = np.arange(len(df_agg)); w = .35
    ax.bar(x - w/2, df_agg["eeg_in_top_mean"], w, yerr=df_agg["eeg_in_top_sd"], capsize=3, color="#4C72B0", label="EEG")
    ax.bar(x + w/2, df_agg["fnirs_in_top_mean"], w, yerr=df_agg["eeg_in_top_sd"], capsize=3, color="#DD8452", label="fNIRS")
    ax.set_xticks(x); ax.set_xticklabels([f"{m}\n(n={n} folds)" for m, n in zip(df_agg["method"], df_agg["n_folds"])], fontsize=9)
    ax.set_ylabel(f"Features in top-{TOP_N} (mean ± SD over folds)"); ax.set_ylim(0, TOP_N); ax.legend(); ax.grid(axis="y", alpha=.3)
    ax.set_title("Modality composition of top features", fontsize=10)
    ax = axes[1]
    err = [blk_summ["mean"] - blk_summ["ci_lo"], blk_summ["ci_hi"] - blk_summ["mean"]]
    ax.bar(blk_summ["block"], blk_summ["mean"], yerr=err, capsize=4, color=["#4C72B0", "#DD8452"])
    ax.axhline(0, color="k", lw=.8); ax.set_ylabel("Macro-AUC drop when block is permuted"); ax.grid(axis="y", alpha=.3)
    ax.set_title(f"Modality-block permutation, RF (no selection)\nmean and 95% CI over {len(recs)} outer folds", fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig5_importance.png"), dpi=200); plt.close()
    return df_agg, ov, blk_summ

def forest_plot(summ, path, title):
    s = summ.sort_values("auc_mean", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 0.36 * len(s) + 1.2))
    ylab = [f"{fs.upper() if fs!='none' else 'None'} + {clf}" for fs, clf in zip(s.fs, s.clf)]
    ax.errorbar(s["auc_mean"], range(len(s)), xerr=[s["auc_mean"] - s["auc_ci_lo"], s["auc_ci_hi"] - s["auc_mean"]],
                fmt="o", color="#333", capsize=3)
    ax.set_yticks(range(len(s))); ax.set_yticklabels(ylab, fontsize=9); ax.axvline(0.5, ls="--", color="gray", lw=.8)
    ax.set_xlabel("Macro AUC (mean, 95% CI)"); ax.set_title(title, fontsize=10); ax.grid(axis="x", alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()

# =============================================================================
# MAIN
# =============================================================================
def main():
    T0 = time.time()
    log("=" * 78); log("EEG-fNIRS AD classification — v2 nested + repeated CV"); log("=" * 78)
    log(f"python {platform.python_version()} | sklearn {sklearn.__version__} | numpy {np.__version__} | pandas {pd.__version__} | scipy {__import__('scipy').__version__}")
    log(f"SEED={SEED}  outer={N_OUTER}x{N_REPEATS}  inner={N_INNER}  K_GRID={K_GRID}  QUICK={QUICK}")
    for c, g in GRIDS.items(): log(f"  grid {c}: {g}")
    X, y, names = load_data()

    # identical outer splits reused by every experiment -> valid paired tests
    rskf = RepeatedStratifiedKFold(n_splits=N_OUTER, n_repeats=N_REPEATS, random_state=SEED)
    splits3 = list(rskf.split(X, y))
    mask2 = y < 2; X2, y2 = X[mask2], y[mask2]
    splits2 = list(RepeatedStratifiedKFold(n_splits=N_OUTER, n_repeats=N_REPEATS, random_state=SEED).split(X2, y2))
    n_tr3, n_te3 = len(splits3[0][0]), len(splits3[0][1])
    n_tr2, n_te2 = len(splits2[0][0]), len(splits2[0][1])

    # ---- Fig.1 : sensitivity curve ----
    sens = exp_sensitivity_curve(X, y, splits3)

    # ---- Table II : three-class nested ----
    log("\n=== THREE-CLASS nested CV (12 configs, tuned + default) ===")
    res3 = nested_cv(X, y, splits3, FS_TYPES, CLF_NAMES, "three_class")
    df3 = pd.DataFrame(res3)
    summ3 = summarize(df3, ["tuned", "fs", "clf"]); summ3.to_csv(os.path.join(OUT_DIR, "table2_three_class.csv"), index=False)
    log("\nTuned:"); log(summ3[summ3.tuned][["fs", "clf", "acc_mean", "f1_mean", "auc_mean", "auc_sd", "auc_ci_lo", "auc_ci_hi", "k_median", "params_mode"]].round(4).to_string(index=False))
    log("\nDefault (untuned, k=50):"); log(summ3[~summ3.tuned][["fs", "clf", "acc_mean", "f1_mean", "auc_mean", "auc_sd", "auc_ci_lo", "auc_ci_hi"]].round(4).to_string(index=False))
    forest_plot(summ3[summ3.tuned], os.path.join(OUT_DIR, "fig2b_forest_three_class.png"), "Three-class: 12 tuned pipelines")
    best3 = summ3[summ3.tuned].sort_values("auc_mean", ascending=False).iloc[0]
    plot_cm(df3, best3.fs, best3.clf, CLASS_NAMES_3, os.path.join(OUT_DIR, "fig2_cm_three_class.png"),
            f"Three-class, {best3.fs}+{best3.clf} (tuned)\ncounts averaged over {N_REPEATS} repeats; row-normalised %")

    # ---- Table III : binary nested ----
    log("\n=== BINARY HC vs MCI nested CV ===")
    res2 = nested_cv(X2, y2, splits2, FS_TYPES, CLF_NAMES, "binary")
    df2 = pd.DataFrame(res2)
    summ2 = summarize(df2, ["tuned", "fs", "clf"]); summ2.to_csv(os.path.join(OUT_DIR, "table3_binary.csv"), index=False)
    log("\nTuned:"); log(summ2[summ2.tuned][["fs", "clf", "acc_mean", "f1_mean", "auc_mean", "auc_sd", "auc_ci_lo", "auc_ci_hi", "k_median", "params_mode"]].round(4).to_string(index=False))
    log("\nDefault (untuned, k=50):"); log(summ2[~summ2.tuned][["fs", "clf", "acc_mean", "f1_mean", "auc_mean", "auc_sd", "auc_ci_lo", "auc_ci_hi"]].round(4).to_string(index=False))
    forest_plot(summ2[summ2.tuned], os.path.join(OUT_DIR, "fig3b_forest_binary.png"), "HC vs MCI: 12 tuned pipelines")
    best2 = summ2[summ2.tuned].sort_values("auc_mean", ascending=False).iloc[0]
    plot_cm(df2, best2.fs, best2.clf, CLASS_NAMES_2, os.path.join(OUT_DIR, "fig3_cm_binary.png"),
            f"HC vs MCI, {best2.fs}+{best2.clf} (tuned)\ncounts averaged over {N_REPEATS} repeats; row-normalised %")

    # ---- Table IV : modality ----
    dfm, summm = exp_modality(X, y, names, splits3)

    # ---- Reviewer #6 : importance ----
    imp_agg, imp_ov, blk = exp_importance(X, y, names, splits3, df3)

    # ---- Reviewer #4 : paired tests ----
    log("\n=== Paired statistical tests (identical outer folds) ===")
    def series(df, fs, clf, tuned=True): return df[(df.fs == fs) & (df.clf == clf) & (df.tuned == tuned)].sort_values("fold_id")["auc"].values
    tests = []
    # three-class family
    fam3 = [("RF-none vs LR-none", series(df3, "none", "RF"), series(df3, "none", "LR")),
            ("RF-none vs LR-anova", series(df3, "none", "RF"), series(df3, "anova", "LR")),
            ("RF-none vs LASSO-none", series(df3, "none", "RF"), series(df3, "none", "LASSO")),
            ("RF-none vs RF-anova", series(df3, "none", "RF"), series(df3, "anova", "RF")),
            ("RF-none vs RF-mi", series(df3, "none", "RF"), series(df3, "mi", "RF"))]
    svm_best = summ3[summ3.tuned & (summ3.clf == "SVM")].sort_values("auc_mean", ascending=False).iloc[0]
    fam3.append((f"RF-none vs SVM-{svm_best.fs} (best SVM)", series(df3, "none", "RF"), series(df3, svm_best.fs, "SVM")))
    fam3.append(("RF-none tuned vs RF-none default", series(df3, "none", "RF"), series(df3, "none", "RF", tuned=False)))
    fam3.append(("SVM-none tuned vs SVM-none default", series(df3, "none", "SVM"), series(df3, "none", "SVM", tuned=False)))
    for name, a, b in fam3: tests.append(dict(task="three_class", comparison=name, **paired_tests(a, b, n_tr3, n_te3)))
    # binary family
    top2 = summ2[summ2.tuned].sort_values("auc_mean", ascending=False).head(3)
    b_series = lambda fs, clf: series(df2, fs, clf)
    for i in range(len(top2)):
        for j in range(i + 1, len(top2)):
            r1, r2 = top2.iloc[i], top2.iloc[j]
            tests.append(dict(task="binary", comparison=f"{r1.clf}-{r1.fs} vs {r2.clf}-{r2.fs}", **paired_tests(b_series(r1.fs, r1.clf), b_series(r2.fs, r2.clf), n_tr2, n_te2)))
    tests.append(dict(task="binary", comparison=f"{top2.iloc[0].clf}-{top2.iloc[0].fs} vs LR-none", **paired_tests(b_series(top2.iloc[0].fs, top2.iloc[0].clf), b_series("none", "LR"), n_tr2, n_te2)))
    # modality family
    def mser(mod, fs, clf): return dfm[(dfm.modality == mod) & (dfm.fs == fs) & (dfm.clf == clf) & dfm.tuned].sort_values("fold_id")["auc"].values
    for fs, clf in [("anova", "LR"), ("none", "RF")]:
        for a, b in [("Multimodal", "EEG-only"), ("Multimodal", "fNIRS-only"), ("EEG-only", "fNIRS-only")]:
            tests.append(dict(task=f"modality_{clf}", comparison=f"{a} vs {b} ({fs}+{clf})", **paired_tests(mser(a, fs, clf), mser(b, fs, clf), n_tr3, n_te3)))
    dft = pd.DataFrame(tests)
    for task, g in dft.groupby("task"):
        dft.loc[g.index, "wilcoxon_p_holm"] = holm(g["wilcoxon_p"].fillna(1).values)
        dft.loc[g.index, "nb_p_holm"] = holm(g["nb_p"].fillna(1).values)
    dft.to_csv(os.path.join(OUT_DIR, "stats_paired_tests.csv"), index=False)
    log(dft[["task", "comparison", "mean_diff", "ci_lo", "ci_hi", "wilcoxon_p", "nb_p", "nb_p_holm"]].round(4).to_string(index=False))

    # ---- selected-k / hyperparameter distributions (reviewer #2) ----
    sel = df3[df3.tuned].groupby(["fs", "clf"]).agg(k_median=("k", "median"), k_min=("k", "min"), k_max=("k", "max"),
                                                    params_mode=("params", lambda s: s.mode().iloc[0]),
                                                    params_mode_freq=("params", lambda s: (s == s.mode().iloc[0]).mean())).reset_index()
    sel.to_csv(os.path.join(OUT_DIR, "selected_hyperparams_three_class.csv"), index=False)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, (fs, c) in enumerate([("anova", "#4C72B0"), ("mi", "#DD8452")]):
        ks = df3[df3.tuned & (df3.fs == fs)]["k"].values
        vals, cnt = np.unique(ks, return_counts=True)
        ax.bar(np.arange(len(K_GRID)) + (i - .5) * .38, [cnt[list(vals).index(k)] if k in vals else 0 for k in K_GRID], .38, color=c, label=fs.upper())
    ax.set_xticks(range(len(K_GRID))); ax.set_xticklabels(K_GRID); ax.set_xlabel("k selected in inner loop"); ax.set_ylabel("count (folds x classifiers)")
    ax.set_title("Distribution of inner-loop-selected k (three-class)", fontsize=10); ax.legend(); ax.grid(axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig1b_selected_k.png"), dpi=200); plt.close()

    log(f"\nTOTAL TIME: {(time.time()-T0)/60:.1f} min"); log(f"Outputs in {OUT_DIR}")

if __name__ == "__main__":
    main()
