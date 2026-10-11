"""
Can a quantum circuit spot a tumor?
Quantum kernel SVM (Qiskit QSVC) vs tuned classical RBF SVMs on Wisconsin Diagnostic Breast Cancer.

Dataset: 569 samples, 30 features; labels 0 = malignant, 1 = benign.
Positive class for F1/recall is malignant (0): a missed cancer is the costly error.

Requirements: qiskit>=2.1, qiskit-machine-learning>=0.9, scikit-learn, scipy, matplotlib, pylatexenc
Run:          python quantum_kernel_breast_cancer.py   (figures + results.json are written to ./outputs)

Single-file layered layout (each layer only uses the ones above it):
  1. Config        2. Data          3. Encoding & quantum kernel   4. Models
  5. Evaluation    6. Visualization 7. Pipeline / entrypoint
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import pauli_feature_map, z_feature_map, zz_feature_map
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from qiskit_machine_learning.utils import algorithm_globals
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

# ============================================================================= 1. CONFIG
RANDOM_STATE = 42
TEST_SIZE = 0.25
CV_FOLDS = 5
CV_REPEATS = 5
MALIGNANT = 0

N_QUBITS = 8
FEATURE_MAPS = ("Z", "ZZ", "Y+ZZ")
REPS_GRID = (1, 2)
BANDWIDTHS = (0.05, 0.1, 0.2, 0.4, 0.7, 1.0)
C_GRID = (0.1, 1.0, 10.0, 100.0)
RBF_GAMMAS_ENCODED = tuple(np.logspace(-3, 1, 9))
RBF_GAMMAS_FULL = tuple(np.logspace(-4, 0, 9))

OUT_DIR = Path(__file__).resolve().parent / "outputs"


# ============================================================================= 2. DATA
def load_and_split():
    data = load_breast_cancer()
    X, y = data.data, data.target
    print("Wisconsin Diagnostic Breast Cancer (sklearn.load_breast_cancer)")
    print(f"  samples={X.shape[0]}, features={X.shape[1]}, "
          f"malignant={(y == MALIGNANT).sum()}, benign={(y == 1).sum()}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    tr = np.arange(len(y_train))
    te = np.arange(len(y_train), len(y_all))
    splits = list(StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(X_train, y_train))
    print(f"  train/test = {len(tr)}/{len(te)} (stratified)\n")
    return X_train, X_test, y_train, y_test, X_all, y_all, tr, te, splits


# ============================================================================= 3. ENCODING & QUANTUM KERNEL
def make_encoder(n_qubits: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=n_qubits, random_state=RANDOM_STATE)),
            ("angle", MinMaxScaler(feature_range=(0.0, np.pi))),
        ]
    )


def build_feature_map(name: str, n_qubits: int, reps: int) -> QuantumCircuit:
    if name == "Z":
        return z_feature_map(n_qubits, reps=reps)
    if name == "ZZ":
        return zz_feature_map(n_qubits, reps=reps, entanglement="linear")
    if name == "Y+ZZ":
        return pauli_feature_map(n_qubits, reps=reps, entanglement="linear", paulis=["Y", "ZZ"])
    raise ValueError(name)


def quantum_gram(fm: str, n_qubits: int, reps: int, X) -> np.ndarray:
    return FidelityStatevectorKernel(feature_map=build_feature_map(fm, n_qubits, reps)).evaluate(X)


# ============================================================================= 4. MODELS
def fit_eval_precomputed(K, y, train_idx, test_idx, C) -> tuple[np.ndarray, np.ndarray]:
    clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
    clf.fit(K[np.ix_(train_idx, train_idx)], y[train_idx])
    K_te = K[np.ix_(test_idx, train_idx)]
    return clf.predict(K_te), -clf.decision_function(K_te)


def cv_f1(K, y, splits, C) -> np.ndarray:
    scores = []
    for tr, va in splits:
        pred, _ = fit_eval_precomputed(K, y, tr, va, C)
        scores.append(f1_score(y[va], pred, pos_label=MALIGNANT))
    return np.array(scores)


def tune_C(K_train, y_train, splits) -> tuple[float, float, dict]:
    per_c = {C: cv_f1(K_train, y_train, splits, C).mean() for C in C_GRID}
    best_C = max(per_c, key=per_c.get)
    return best_C, per_c[best_C], per_c


def tune_rbf(X_all, y_train, train_idx, splits, gammas) -> dict:
    best = {"cv_f1": -1.0}
    for g in gammas:
        K = rbf_kernel(X_all, gamma=g)
        C, score, _ = tune_C(K[np.ix_(train_idx, train_idx)], y_train, splits)
        if score > best["cv_f1"]:
            best = {"gamma": float(g), "C": C, "cv_f1": score, "K": K}
    return best


def select_quantum_kernel(Z_all, y_train, tr, splits):
    """Grid search over feature map x reps x bandwidth x C (5-fold CV on the training set)."""
    grid, best = {}, {"cv_f1": -1.0}
    for fm in FEATURE_MAPS:
        for reps in REPS_GRID:
            grid[(fm, reps)] = {}
            for bw in BANDWIDTHS:
                K = quantum_gram(fm, N_QUBITS, reps, Z_all * bw)
                _, _, per_c = tune_C(K[np.ix_(tr, tr)], y_train, splits)
                for C, s in per_c.items():
                    grid[(fm, reps)][(bw, C)] = s
                    if s > best["cv_f1"]:
                        best = {"fm": fm, "reps": reps, "bw": bw, "C": C, "cv_f1": s, "K": K}
            top = max(grid[(fm, reps)].items(), key=lambda kv: kv[1])
            print(f"  {fm:5s} reps={reps}: best CV F1(m)={top[1]:.3f} at bandwidth={top[0][0]:g}, C={top[0][1]:g}")
    print(f"  -> selected {best['fm']} reps={best['reps']} bandwidth={best['bw']:g} "
          f"C={best['C']:g} CV F1(m)={best['cv_f1']:.3f}")
    return grid, best


def fit_qsvc(cfg, Z_all, y_train, tr, te):
    kernel = FidelityStatevectorKernel(feature_map=build_feature_map(cfg["fm"], N_QUBITS, cfg["reps"]))
    qsvc = QSVC(quantum_kernel=kernel, C=cfg["C"], class_weight="balanced")
    t0 = time.time()
    qsvc.fit(Z_all[tr] * cfg["bw"], y_train)
    pred = qsvc.predict(Z_all[te] * cfg["bw"])
    score = -qsvc.decision_function(Z_all[te] * cfg["bw"])
    return pred, score, time.time() - t0, len(qsvc.support_)


# ============================================================================= 5. EVALUATION
def compute_metrics(y_true, y_pred, malignant_score) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_malignant": f1_score(y_true, y_pred, pos_label=MALIGNANT),
        "recall_malignant": recall_score(y_true, y_pred, pos_label=MALIGNANT),
        "precision_malignant": precision_score(y_true, y_pred, pos_label=MALIGNANT),
        "roc_auc": roc_auc_score((y_true == MALIGNANT).astype(int), malignant_score),
    }


def format_metrics(m: dict) -> str:
    return (f"acc={m['accuracy']:.3f}  F1(m)={m['f1_malignant']:.3f}  "
            f"recall(m)={m['recall_malignant']:.3f}  AUC={m['roc_auc']:.3f}")


def centered_alignment(K, y) -> float:
    n = len(y)
    yy = np.where(y == MALIGNANT, 1.0, -1.0)
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Y = np.outer(yy, yy)
    return float(np.sum(Kc * Y) / (np.linalg.norm(Kc) * np.linalg.norm(Y)))


def corrected_paired_ttest(diffs, n_train, n_test) -> tuple[float, float]:
    k = len(diffs)
    var = np.var(diffs, ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = diffs.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return float(t), float(p)


# ============================================================================= 6. VISUALIZATION
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8985"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
QUANTUM = "#2a78d6"
CLASSICAL = "#eb6834"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.frameon": False, "lines.linewidth": 2,
        "lines.markersize": 6, "font.size": 9, "text.color": INK,
    })


def save(fig, name: str) -> None:
    fig.savefig(OUT_DIR / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved outputs/{name}")


def entity_color(name: str) -> str:
    if name.startswith("QSVC"):
        return QUANTUM
    if "same" in name:
        return CLASSICAL
    return INK_3


def plot_circuit(fm: str, reps: int, label: str, filename: str) -> None:
    circuit = build_feature_map(fm, N_QUBITS, reps).decompose()
    try:
        fig = circuit.draw(output="mpl", fold=40, style="clifford")
        fig.suptitle(
            f"{label}: {fm}, {N_QUBITS} qubits, reps={reps}  "
            f"(depth {circuit.depth()}, {circuit.num_nonlocal_gates()} two-qubit gates)",
            x=0.02, ha="left", fontsize=11, color=INK,
        )
        save(fig, f"{filename}.png")
    except Exception as exc:
        (OUT_DIR / f"{filename}.txt").write_text(str(circuit.draw(output="text")))
        print(f"  mpl circuit drawer unavailable ({exc}); wrote text diagram instead")


def plot_model_selection(grid: dict, best: dict) -> None:
    fig, axes = plt.subplots(len(REPS_GRID), len(FEATURE_MAPS), figsize=(13, 3.2 * len(REPS_GRID)),
                             squeeze=False)
    values = np.array([v for d in grid.values() for v in d.values()])
    vmin, vmax = values.min(), values.max()
    for i, reps in enumerate(REPS_GRID):
        for j, fm in enumerate(FEATURE_MAPS):
            ax = axes[i, j]
            M = np.array([[grid[(fm, reps)][(bw, C)] for C in C_GRID] for bw in BANDWIDTHS])
            ax.imshow(M, cmap=SEQ, vmin=vmin, vmax=vmax, aspect="auto")
            for (r, c), v in np.ndenumerate(M):
                is_best = (fm, reps, BANDWIDTHS[r], C_GRID[c]) == (best["fm"], best["reps"], best["bw"], best["C"])
                ax.text(c, r, f"{v:.3f}", ha="center", va="center", fontsize=7,
                        color=SURFACE if v > (vmin + vmax) / 2 else INK,
                        fontweight="bold" if is_best else "normal")
                if is_best:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                               edgecolor=CLASSICAL, linewidth=2))
            ax.set_xticks(range(len(C_GRID)), [f"{C:g}" for C in C_GRID])
            ax.set_yticks(range(len(BANDWIDTHS)), [f"{b:g}" for b in BANDWIDTHS])
            ax.set_xlabel("SVM C")
            ax.set_ylabel("bandwidth")
            ax.set_title(f"{fm} map, reps={reps}")
            ax.grid(False)
    fig.suptitle("QSVC model selection: 5-fold CV F1(malignant) on training set (best boxed)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "03_qsvc_model_selection.png")


def plot_leaderboard(test_results: dict, n_test: int) -> None:
    names = sorted(test_results, key=lambda n: test_results[n]["f1_malignant"])
    metrics = (("accuracy", "Accuracy"), ("f1_malignant", "F1 (malignant)"),
               ("recall_malignant", "Recall (malignant)"), ("roc_auc", "ROC AUC"))
    fig, axes = plt.subplots(1, len(metrics), figsize=(17, 0.42 * len(names) + 1.6), sharey=True)
    for ax, (key, title) in zip(axes, metrics):
        vals = [test_results[n][key] for n in names]
        colors = [entity_color(n) for n in names]
        ax.hlines(range(len(names)), min(vals) - 0.01, vals, color=GRID, linewidth=1)
        ax.scatter(vals, range(len(names)), color=colors, s=60, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        for i, v in enumerate(vals):
            ax.text(v, i + 0.28, f"{v:.3f}", ha="center", fontsize=7, color=INK_2)
        ax.set_title(title)
        ax.set_xlim(min(vals) - 0.01, 1.003)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(names)), names)
    fig.suptitle(f"Held-out test set ({n_test} patients): every model, same split", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "08_test_leaderboard.png")


def plot_roc_confusion(y_test, curves: dict, preds: dict) -> None:
    fig, axes = plt.subplots(1, 1 + len(preds), figsize=(5 * (1 + len(preds)), 4.4))
    ax = axes[0]
    y_bin = (y_test == MALIGNANT).astype(int)
    for name, score in curves.items():
        fpr, tpr, _ = roc_curve(y_bin, score)
        ax.plot(fpr, tpr, color=entity_color(name), label=f"{name} (AUC {roc_auc_score(y_bin, score):.3f})")
    ax.plot([0, 1], [0, 1], color=GRID, linestyle="--", linewidth=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate (malignant caught)")
    ax.set_title("ROC on held-out test set")
    ax.legend(loc="lower right", fontsize=7)

    for ax, (name, pred) in zip(axes[1:], preds.items()):
        cm = confusion_matrix(y_test, pred, labels=[MALIGNANT, 1])
        ax.imshow(cm, cmap=SEQ, vmin=0, vmax=cm.max())
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, str(v), ha="center", va="center", fontsize=14,
                    color=SURFACE if v > cm.max() / 2 else INK)
        ax.set_xticks([0, 1], ["malignant", "benign"])
        ax.set_yticks([0, 1], ["malignant", "benign"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_title(f"{name}\nmissed cancers: {cm[0, 1]}", fontsize=10)
        ax.grid(False)
    fig.tight_layout()
    save(fig, "09_roc_confusion.png")


def plot_significance(diffs: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 0.9 * len(diffs) + 1.4))
    rng = np.random.default_rng(RANDOM_STATE)
    for i, (name, d) in enumerate(diffs.items()):
        jitter = rng.uniform(-0.12, 0.12, len(d["diffs"]))
        ax.scatter(d["diffs"], i + jitter, s=14, color=INK_3, alpha=0.8, edgecolor="none")
        ax.scatter([np.mean(d["diffs"])], [i], s=90, color=QUANTUM, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.text(max(d["diffs"]) + 0.004, i, f"mean {np.mean(d['diffs']):+.4f}, p={d['p']:.3f}",
                va="center", color=INK_2, fontsize=8)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(range(len(diffs)), list(diffs))
    ax.invert_yaxis()
    ax.set_xlabel("F1(malignant) difference per fold  (> 0 favours the QSVC)")
    ax.set_title(f"Paired {CV_REPEATS}x{CV_FOLDS} repeated CV, corrected resampled t-test")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.25)
    fig.tight_layout()
    save(fig, "11_paired_significance.png")


# ============================================================================= 7. PIPELINE
def main() -> None:
    t_start = time.time()
    set_style()
    OUT_DIR.mkdir(exist_ok=True)
    algorithm_globals.random_seed = RANDOM_STATE

    X_train, X_test, y_train, y_test, X_all, y_all, tr, te, splits = load_and_split()
    test_results, test_scores, test_preds = {}, {}, {}

    def record(name, pred, score):
        test_results[name] = compute_metrics(y_test, pred, score)
        test_scores[name], test_preds[name] = score, pred

    # --- classical reference: tuned RBF SVC on all 30 features
    print("[1/4] Classical reference: tuned RBF SVC on all 30 features")
    X_all_std = StandardScaler().fit(X_train).transform(X_all)
    full_rbf = tune_rbf(X_all_std, y_train, tr, splits, RBF_GAMMAS_FULL)
    name_full = "SVC RBF tuned (30 feat)"
    record(name_full, *fit_eval_precomputed(full_rbf["K"], y_all, tr, te, full_rbf["C"]))
    print(f"  {name_full:28s} CV F1(m)={full_rbf['cv_f1']:.3f} | test {format_metrics(test_results[name_full])}"
          f"  (gamma={full_rbf['gamma']:.4g}, C={full_rbf['C']:g})")

    # --- quantum model selection
    print(f"\n[2/4] Quantum encoding (StandardScaler -> PCA({N_QUBITS}) -> angles) and QSVC model selection")
    Z_all = make_encoder(N_QUBITS).fit(X_train).transform(X_all)
    grid, best_q = select_quantum_kernel(Z_all, y_train, tr, splits)
    plot_circuit(best_q["fm"], best_q["reps"], "Selected quantum feature map", "02_feature_map_circuit")
    plot_model_selection(grid, best_q)

    # --- matched classical baseline + final test evaluation
    print("\n[3/4] Fair head-to-head on identical encoded features + test evaluation")
    matched = tune_rbf(Z_all, y_train, tr, splits, RBF_GAMMAS_ENCODED)
    print(f"  RBF same features: CV F1(m)={matched['cv_f1']:.3f} (gamma={matched['gamma']:.4g}, C={matched['C']:g})")

    name_q = f"QSVC {best_q['fm']} ({N_QUBITS} qubits)"
    q_pred, q_score, q_time, n_sv = fit_qsvc(best_q, Z_all, y_train, tr, te)
    record(name_q, q_pred, q_score)
    print(f"  {name_q:28s} test {format_metrics(test_results[name_q])}  "
          f"(fit+predict {q_time:.1f}s, {n_sv} support vectors)")

    name_m = f"SVC RBF same {N_QUBITS} feat"
    record(name_m, *fit_eval_precomputed(matched["K"], y_all, tr, te, matched["C"]))
    print(f"  {name_m:28s} test {format_metrics(test_results[name_m])}")

    # --- paired significance
    print(f"\n[4/4] Paired significance: {CV_REPEATS}x{CV_FOLDS} repeated CV on training set")
    rep_splits = list(RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS,
                                              random_state=RANDOM_STATE).split(X_train, y_train))
    rep = {
        "quantum": cv_f1(best_q["K"][np.ix_(tr, tr)], y_train, rep_splits, best_q["C"]),
        "matched": cv_f1(matched["K"][np.ix_(tr, tr)], y_train, rep_splits, matched["C"]),
        "full": cv_f1(full_rbf["K"][np.ix_(tr, tr)], y_train, rep_splits, full_rbf["C"]),
    }
    n_val = len(tr) // CV_FOLDS
    comparisons = {
        "QSVC vs RBF (same features)": rep["quantum"] - rep["matched"],
        "QSVC vs RBF (30 features)": rep["quantum"] - rep["full"],
    }
    significance = {}
    for label, d in comparisons.items():
        t, p = corrected_paired_ttest(d, len(tr) - n_val, n_val)
        significance[label] = {"diffs": d, "mean": float(d.mean()), "t": t, "p": p,
                               "wins": int((d > 0).sum()), "ties": int((d == 0).sum()),
                               "losses": int((d < 0).sum())}
        s = significance[label]
        print(f"  {label:30s} mean dF1={d.mean():+.4f}  t={t:+.2f}  p={p:.3f}  "
              f"W/T/L={s['wins']}/{s['ties']}/{s['losses']}")
    plot_significance(significance)

    alignment = {
        "Quantum (QSVC)": centered_alignment(best_q["K"][np.ix_(tr, tr)], y_train),
        "RBF, same features": centered_alignment(matched["K"][np.ix_(tr, tr)], y_train),
        "RBF, 30 features": centered_alignment(full_rbf["K"][np.ix_(tr, tr)], y_train),
    }
    for k, v in alignment.items():
        print(f"  {k:20s} centered alignment={v:.3f}")

    # --- reporting
    plot_leaderboard(test_results, len(te))
    plot_roc_confusion(y_test,
                       {k: test_scores[k] for k in (name_full, name_m, name_q)},
                       {k: test_preds[k] for k in (name_q, name_m)})

    verdict = {
        "qsvc_vs_matched_rbf_test_dF1": test_results[name_q]["f1_malignant"] - test_results[name_m]["f1_malignant"],
        "qsvc_vs_full_rbf_test_dF1": test_results[name_q]["f1_malignant"] - test_results[name_full]["f1_malignant"],
        "missed_cancers": {k: int(confusion_matrix(y_test, test_preds[k], labels=[MALIGNANT, 1])[0, 1])
                           for k in (name_q, name_m, name_full)},
    }
    results = {
        "config": {"n_qubits": N_QUBITS, "feature_maps": FEATURE_MAPS, "reps_grid": REPS_GRID,
                   "bandwidths": BANDWIDTHS, "C_grid": C_GRID, "random_state": RANDOM_STATE},
        "selected_quantum": {k: v for k, v in best_q.items() if k != "K"},
        "selected_matched_rbf": {k: v for k, v in matched.items() if k != "K"},
        "selected_full_rbf": {k: v for k, v in full_rbf.items() if k != "K"},
        "test_results": test_results,
        "repeated_cv_mean_f1": {k: float(v.mean()) for k, v in rep.items()},
        "significance": {k: {kk: vv for kk, vv in v.items() if kk != "diffs"} for k, v in significance.items()},
        "kernel_alignment": alignment,
        "verdict": verdict,
        "runtime_seconds": time.time() - t_start,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=float))

    print("\n================ RESULTS (held-out test set, positive class = malignant) ================")
    for name in sorted(test_results, key=lambda k: -test_results[k]["f1_malignant"]):
        print(f"  {name:28s} {format_metrics(test_results[name])}")
    sig_m = significance["QSVC vs RBF (same features)"]
    print("\n================ VERDICT ================")
    print(f"  QSVC vs RBF, identical {N_QUBITS} features: dF1 {verdict['qsvc_vs_matched_rbf_test_dF1']:+.3f} test | "
          f"{sig_m['mean']:+.4f} repeated CV (p={sig_m['p']:.3f})")
    print(f"  QSVC vs RBF, 30 features: dF1 {verdict['qsvc_vs_full_rbf_test_dF1']:+.3f} test")
    print(f"  Missed cancers on test set: {verdict['missed_cancers']}")
    print(f"\n  Total runtime {time.time() - t_start:.0f}s. Figures + results.json in {OUT_DIR}")


if __name__ == "__main__":
    main()
