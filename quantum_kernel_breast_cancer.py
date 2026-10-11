"""
Can a quantum circuit help spot a tumor?
This script compares a quantum-kernel SVM (Qiskit QSVC) against tuned classical RBF SVM baselines
on the Wisconsin Diagnostic Breast Cancer dataset.

Dataset: 569 samples, 30 features; labels 0 = malignant, 1 = benign.
For F1/recall, the positive class is malignant (0), since missed cancers are the most costly errors.

Requirements: qiskit>=2.1, qiskit-machine-learning>=0.9, scikit-learn, scipy, matplotlib, pylatexenc
Run:          python quantum_kernel_breast_cancer.py   (figures + results.json are written to ./outputs)

Single-file layered layout (each layer only depends on the ones above it):
    1. Config        2. Data          3. Encoding & quantum kernel   4. Models
    5. Evaluation    6. Visualization 7. Pipeline / entrypoint
"""

import json
import time
from dataclasses import dataclass
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

# Set to "ionq" in Qollab to run the reduced, shot-based backend experiment.
RUN_MODE = "local"
IONQ_SAMPLE_COUNT = 12
IONQ_SHOTS = 64

PROJECT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUT_DIR = PROJECT_DIR / "outputs"


@dataclass
class SplitArtifacts:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    X_all: np.ndarray
    y_all: np.ndarray
    train_idx: np.ndarray
    test_idx: np.ndarray
    cv_splits: list


# ============================================================================= 2. DATA
def load_breast_cancer_data() -> tuple[np.ndarray, np.ndarray]:
    data = load_breast_cancer()
    X, y = data.data, data.target
    print("Wisconsin Diagnostic Breast Cancer (sklearn.load_breast_cancer)")
    print(f"  samples={X.shape[0]}, features={X.shape[1]}, "
          f"malignant={(y == MALIGNANT).sum()}, benign={(y == 1).sum()}")
    return X, y


def prepare_test_and_validation_data(X: np.ndarray, y: np.ndarray) -> SplitArtifacts:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    tr = np.arange(len(y_train))
    te = np.arange(len(y_train), len(y_all))
    splits = list(StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(X_train, y_train))
    print(f"  train/test = {len(tr)}/{len(te)} (stratified)\n")
    return SplitArtifacts(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_all=X_all,
        y_all=y_all,
        train_idx=tr,
        test_idx=te,
        cv_splits=splits,
    )


# ============================================================================= 3. QUANTUM CONCEPT GROUPING
# Superposition: continuous features are encoded into qubit rotation angles.
def make_encoder(n_qubits: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=n_qubits, random_state=RANDOM_STATE)),
            ("angle", MinMaxScaler(feature_range=(0.0, np.pi))),
        ]
    )


def apply_superposition_encoding(X_train: np.ndarray, X_all: np.ndarray, n_qubits: int) -> np.ndarray:
    return make_encoder(n_qubits).fit(X_train).transform(X_all)


# Entanglement: ZZ and Y+ZZ maps add two-qubit interactions between neighboring qubits.
def build_feature_map(name: str, n_qubits: int, reps: int) -> QuantumCircuit:
    if name == "Z":
        return z_feature_map(n_qubits, reps=reps)
    if name == "ZZ":
        return zz_feature_map(n_qubits, reps=reps, entanglement="linear")
    if name == "Y+ZZ":
        return pauli_feature_map(n_qubits, reps=reps, entanglement="linear", paulis=["Y", "ZZ"])
    raise ValueError(name)


# Interference: fidelity kernel computes overlap between encoded quantum states.
def quantum_gram(fm: str, n_qubits: int, reps: int, X) -> np.ndarray:
    return FidelityStatevectorKernel(feature_map=build_feature_map(fm, n_qubits, reps)).evaluate(X)


def build_fidelity_circuit(feature_map: QuantumCircuit, left: np.ndarray,
                           right: np.ndarray) -> QuantumCircuit:
    left_state = feature_map.assign_parameters(dict(zip(feature_map.parameters, left)))
    right_state = feature_map.assign_parameters(dict(zip(feature_map.parameters, right)))
    circuit = QuantumCircuit(feature_map.num_qubits, feature_map.num_qubits)
    circuit.compose(left_state, inplace=True)
    circuit.compose(right_state.inverse(), inplace=True)
    circuit.measure(range(feature_map.num_qubits), range(feature_map.num_qubits))
    return circuit


def backend_quantum_gram(backend, fm: str, n_qubits: int, reps: int,
                         X_train: np.ndarray, X_test: np.ndarray,
                         shots: int = IONQ_SHOTS) -> tuple[np.ndarray, np.ndarray]:
    """Compute train/train and test/train fidelity kernels in one backend job."""
    if shots < 1:
        raise ValueError("shots must be positive")

    feature_map = build_feature_map(fm, n_qubits, reps)
    circuits, pairs = [], []

    def add_pair(left: np.ndarray, right: np.ndarray, split: str, row: int, col: int) -> None:
        circuits.append(build_fidelity_circuit(feature_map, left, right))
        pairs.append((split, row, col))

    K_train = np.eye(len(X_train), dtype=float)
    K_test = np.empty((len(X_test), len(X_train)), dtype=float)
    for row in range(len(X_train)):
        for col in range(row):
            add_pair(X_train[row], X_train[col], "train", row, col)
    for row in range(len(X_test)):
        for col in range(len(X_train)):
            add_pair(X_test[row], X_train[col], "test", row, col)

    if not circuits:
        return K_train, K_test
    print(f"Submitting one backend job with {len(circuits)} circuits")
    job = backend.run(circuits, shots=shots)
    result = job.result()
    for index, (split, row, col) in enumerate(pairs):
        counts = result.get_counts(index)
        zero_count = sum(count for bitstring, count in counts.items()
                         if bitstring.replace(" ", "").count("1") == 0)
        fidelity = zero_count / shots
        if split == "train":
            K_train[row, col] = K_train[col, row] = fidelity
        else:
            K_test[row, col] = fidelity
    return K_train, K_test


def run_ionq_demo() -> None:
    qollab_backend = globals().get("backend")
    if qollab_backend is None:
        raise RuntimeError("IonQ mode requires Qollab's injected `backend`; select an IonQ simulator.")

    X, y = load_breast_cancer_data()
    X_small, _, y_small, _ = train_test_split(
        X, y, train_size=IONQ_SAMPLE_COUNT, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_small, y_small, test_size=0.25, stratify=y_small, random_state=RANDOM_STATE
    )
    encoder = make_encoder(N_QUBITS)
    encoded = encoder.fit(X_train).transform(np.vstack([X_train, X_test]))
    Z_train = encoded[:len(X_train)]
    Z_test = encoded[len(X_train):]

    feature_map = build_feature_map("Z", N_QUBITS, 1)
    circuit = build_fidelity_circuit(feature_map, Z_train[0], Z_train[1])
    print(f"IonQ backend run: one 8-qubit fidelity circuit, {IONQ_SHOTS} shots")
    print("Submitting one circuit because Qollab's IonQ endpoint accepts ionq.circuit.v1, not circuit batches.")
    job = qollab_backend.run(circuit, shots=IONQ_SHOTS)
    counts = job.result().get_counts()
    zero_count = sum(count for bitstring, count in counts.items()
                     if bitstring.replace(" ", "").count("1") == 0)
    fidelity = zero_count / IONQ_SHOTS
    print(f"Measured feature-map fidelity for one patient pair: {fidelity:.3f}")
    print("This confirms IonQ simulator execution; classification metrics come from local mode.")
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "ionq_results.json").write_text(json.dumps({
        "backend": str(qollab_backend),
        "mode": "ionq_single_circuit_backend_check",
        "samples_used_to_fit_encoder": IONQ_SAMPLE_COUNT,
        "shots": IONQ_SHOTS,
        "feature_map": "Z",
        "reps": 1,
        "measured_pair_fidelity": fidelity,
        "counts": counts,
    }, indent=2, default=float))


# Decoherence-inspired control: bandwidth scales encoded angles to control effective phase spread.
def decoherence_bandwidth_scale(encoded_features: np.ndarray, bandwidth: float) -> np.ndarray:
    return encoded_features * bandwidth


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
                K = quantum_gram(fm, N_QUBITS, reps, decoherence_bandwidth_scale(Z_all, bw))
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
    z_train = decoherence_bandwidth_scale(Z_all[tr], cfg["bw"])
    z_test = decoherence_bandwidth_scale(Z_all[te], cfg["bw"])
    qsvc.fit(z_train, y_train)
    pred = qsvc.predict(z_test)
    score = -qsvc.decision_function(z_test)
    return pred, score, time.time() - t0, len(qsvc.support_)


# ============================================================================= 5. EVALUATION
def compute_metrics(y_true, y_pred, malignant_score) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_malignant": f1_score(y_true, y_pred, pos_label=MALIGNANT),
        "recall_malignant": recall_score(y_true, y_pred, pos_label=MALIGNANT),
        "precision_malignant": precision_score(y_true, y_pred, pos_label=MALIGNANT, zero_division=0),
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
    denom = np.linalg.norm(Kc) * np.linalg.norm(Y)
    if denom == 0:
        return 0.0
    return float(np.sum(Kc * Y) / denom)


def corrected_paired_ttest(diffs, n_train, n_test) -> tuple[float, float]:
    diffs = np.asarray(diffs, dtype=float)
    k = len(diffs)
    if k < 2 or n_train <= 0 or n_test < 0:
        return 0.0, 1.0
    var = np.var(diffs, ddof=1)
    if var == 0 or not np.isfinite(var):
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
    save(fig, "02_qsvc_model_selection.png")


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
    save(fig, "04_test_leaderboard.png")


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
    save(fig, "05_roc_confusion.png")


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
    save(fig, "03_paired_significance.png")


# ============================================================================= 7. PIPELINE
def run_classical_baseline_all_features(split: SplitArtifacts):
    X_all_std = StandardScaler().fit(split.X_train).transform(split.X_all)
    best = tune_rbf(X_all_std, split.y_train, split.train_idx, split.cv_splits, RBF_GAMMAS_FULL)
    return best, fit_eval_precomputed(best["K"], split.y_all, split.train_idx, split.test_idx, best["C"])


def run_quantum_pipeline(split: SplitArtifacts):
    encoded = apply_superposition_encoding(split.X_train, split.X_all, N_QUBITS)
    grid, best_q = select_quantum_kernel(encoded, split.y_train, split.train_idx, split.cv_splits)
    q_pred, q_score, q_time, n_sv = fit_qsvc(best_q, encoded, split.y_train, split.train_idx, split.test_idx)
    return encoded, grid, best_q, (q_pred, q_score, q_time, n_sv)


def run_classical_baseline_same_encoded_features(split: SplitArtifacts, encoded_features: np.ndarray):
    best = tune_rbf(encoded_features, split.y_train, split.train_idx, split.cv_splits, RBF_GAMMAS_ENCODED)
    pred, score = fit_eval_precomputed(best["K"], split.y_all, split.train_idx, split.test_idx, best["C"])
    return best, (pred, score)


def main() -> None:
    if RUN_MODE == "ionq":
        run_ionq_demo()
        return

    t_start = time.time()
    set_style()
    OUT_DIR.mkdir(exist_ok=True)
    algorithm_globals.random_seed = RANDOM_STATE

    # Step 1 and Step 2: load data and prepare train/test + validation folds.
    print("[1/8] Load data")
    X, y = load_breast_cancer_data()
    print("\n[2/8] Prepare train/test split and validation folds")
    split = prepare_test_and_validation_data(X, y)

    test_results, test_scores, test_preds = {}, {}, {}

    def record(name, pred, score):
        test_results[name] = compute_metrics(split.y_test, pred, score)
        test_scores[name], test_preds[name] = score, pred

    # Step 3: quantum concept grouping in code.
    print(f"\n[3/8] Quantum concept groups used in this pipeline")
    print("  superposition: angle encoding with StandardScaler -> PCA -> MinMax(0, pi)")
    print("  entanglement: ZZ / Y+ZZ feature maps with linear two-qubit interactions")
    print("  interference: fidelity kernel overlap creates the quantum similarity matrix")
    print("  decoherence-inspired control: bandwidth scaling on encoded angles")

    # Step 4 and Step 5: run logic on existing classical model + metrics.
    print("\n[4/8] Apply logic on existing model: tuned classical RBF SVM (30 features)")
    full_rbf, (full_pred, full_score) = run_classical_baseline_all_features(split)
    name_full = "SVC RBF tuned (30 feat)"
    record(name_full, full_pred, full_score)
    print("[5/8] Metrics score and other for existing model")
    print(f"  {name_full:28s} CV F1(m)={full_rbf['cv_f1']:.3f} | test {format_metrics(test_results[name_full])}"
                f"  (gamma={full_rbf['gamma']:.4g}, C={full_rbf['C']:g})")

    # Step 6 and Step 7: create quantum SVM and report metrics.
    print(f"\n[6/8] Create Quantum SVM and apply quantum models")
    Z_all, grid, best_q, (q_pred, q_score, q_time, n_sv) = run_quantum_pipeline(split)
    plot_circuit(best_q["fm"], best_q["reps"], "Selected quantum feature map", "01_feature_map_circuit")
    plot_model_selection(grid, best_q)

    print("\n[7/8] Metrics score and other for quantum model + matched classical baseline")
    matched, (matched_pred, matched_score) = run_classical_baseline_same_encoded_features(split, Z_all)
    print(f"  RBF same features: CV F1(m)={matched['cv_f1']:.3f} (gamma={matched['gamma']:.4g}, C={matched['C']:g})")

    name_q = f"QSVC {best_q['fm']} ({N_QUBITS} qubits)"
    record(name_q, q_pred, q_score)
    print(f"  {name_q:28s} test {format_metrics(test_results[name_q])}  "
          f"(fit+predict {q_time:.1f}s, {n_sv} support vectors)")

    name_m = f"SVC RBF same {N_QUBITS} feat"
    record(name_m, matched_pred, matched_score)
    print(f"  {name_m:28s} test {format_metrics(test_results[name_m])}")

    # Step 8: final comparison and significance.
    print(f"\n[8/8] Comparison: paired significance ({CV_REPEATS}x{CV_FOLDS} repeated CV on training set)")
    rep_splits = list(RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS,
                                              random_state=RANDOM_STATE).split(split.X_train, split.y_train))
    rep = {
        "quantum": cv_f1(best_q["K"][np.ix_(split.train_idx, split.train_idx)], split.y_train, rep_splits, best_q["C"]),
        "matched": cv_f1(matched["K"][np.ix_(split.train_idx, split.train_idx)], split.y_train, rep_splits, matched["C"]),
        "full": cv_f1(full_rbf["K"][np.ix_(split.train_idx, split.train_idx)], split.y_train, rep_splits, full_rbf["C"]),
    }
    n_val = len(split.train_idx) // CV_FOLDS
    comparisons = {
        "QSVC vs RBF (same features)": rep["quantum"] - rep["matched"],
        "QSVC vs RBF (30 features)": rep["quantum"] - rep["full"],
    }
    significance = {}
    for label, d in comparisons.items():
        t, p = corrected_paired_ttest(d, len(split.train_idx) - n_val, n_val)
        significance[label] = {"diffs": d, "mean": float(d.mean()), "t": t, "p": p,
                               "wins": int((d > 0).sum()), "ties": int((d == 0).sum()),
                               "losses": int((d < 0).sum())}
        s = significance[label]
        print(f"  {label:30s} mean dF1={d.mean():+.4f}  t={t:+.2f}  p={p:.3f}  "
              f"W/T/L={s['wins']}/{s['ties']}/{s['losses']}")
    plot_significance(significance)

    alignment = {
        "Quantum (QSVC)": centered_alignment(best_q["K"][np.ix_(split.train_idx, split.train_idx)], split.y_train),
        "RBF, same features": centered_alignment(matched["K"][np.ix_(split.train_idx, split.train_idx)], split.y_train),
        "RBF, 30 features": centered_alignment(full_rbf["K"][np.ix_(split.train_idx, split.train_idx)], split.y_train),
    }
    for k, v in alignment.items():
        print(f"  {k:20s} centered alignment={v:.3f}")

    # --- reporting
    plot_leaderboard(test_results, len(split.test_idx))
    plot_roc_confusion(split.y_test,
                       {k: test_scores[k] for k in (name_full, name_m, name_q)},
                       {k: test_preds[k] for k in (name_q, name_m)})

    verdict = {
        "qsvc_vs_matched_rbf_test_dF1": test_results[name_q]["f1_malignant"] - test_results[name_m]["f1_malignant"],
        "qsvc_vs_full_rbf_test_dF1": test_results[name_q]["f1_malignant"] - test_results[name_full]["f1_malignant"],
        "missed_cancers": {k: int(confusion_matrix(split.y_test, test_preds[k], labels=[MALIGNANT, 1])[0, 1])
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


if __name__ == "__main__" and "backend" not in globals():
    main()
