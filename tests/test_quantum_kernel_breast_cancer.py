import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import quantum_kernel_breast_cancer as q


def test_load_breast_cancer_data(monkeypatch):
    fake = types.SimpleNamespace(
        data=np.array([[1.0, 2.0], [3.0, 4.0]]),
        target=np.array([0, 1]),
    )
    monkeypatch.setattr(q, "load_breast_cancer", lambda: fake)

    X, y = q.load_breast_cancer_data()

    assert X.shape == (2, 2)
    assert y.tolist() == [0, 1]


def test_prepare_test_and_validation_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    y = np.array([0, 1] * 20)

    split = q.prepare_test_and_validation_data(X, y)

    assert split.X_all.shape[0] == 40
    assert split.y_all.shape[0] == 40
    assert len(split.train_idx) + len(split.test_idx) == 40
    assert len(split.cv_splits) == q.CV_FOLDS


def test_make_encoder_and_apply_superposition_encoding():
    rng = np.random.default_rng(1)
    X_train = rng.normal(size=(24, 5))
    X_all = rng.normal(size=(30, 5))

    encoded = q.apply_superposition_encoding(X_train, X_all, n_qubits=3)

    assert encoded.shape == (30, 3)
    assert np.all(np.isfinite(encoded))


def test_build_feature_map_variants_and_invalid():
    for name in ("Z", "ZZ", "Y+ZZ"):
        fm = q.build_feature_map(name, n_qubits=2, reps=1)
        assert fm.num_qubits == 2

    with pytest.raises(ValueError):
        q.build_feature_map("BAD", n_qubits=2, reps=1)


def test_quantum_gram(monkeypatch):
    class DummyKernel:
        def __init__(self, feature_map):
            self.feature_map = feature_map

        def evaluate(self, X):
            n = len(X)
            return np.eye(n)

    monkeypatch.setattr(q, "FidelityStatevectorKernel", DummyKernel)

    K = q.quantum_gram("Z", n_qubits=2, reps=1, X=np.array([[0.0], [1.0]]))
    assert K.shape == (2, 2)
    assert np.allclose(K, np.eye(2))


def test_decoherence_bandwidth_scale():
    x = np.array([[1.0, 2.0]])
    out = q.decoherence_bandwidth_scale(x, 0.5)
    assert np.allclose(out, np.array([[0.5, 1.0]]))


def test_fit_eval_precomputed_and_cv_f1():
    K = np.array(
        [
            [1.0, 0.9, 0.2, 0.1],
            [0.9, 1.0, 0.1, 0.2],
            [0.2, 0.1, 1.0, 0.9],
            [0.1, 0.2, 0.9, 1.0],
        ]
    )
    y = np.array([0, 0, 1, 1])
    pred, score = q.fit_eval_precomputed(K, y, train_idx=np.array([0, 2]), test_idx=np.array([1, 3]), C=1.0)

    assert pred.shape == (2,)
    assert score.shape == (2,)

    splits = [(np.array([0, 2]), np.array([1, 3])), (np.array([1, 3]), np.array([0, 2]))]
    scores = q.cv_f1(K, y, splits, C=1.0)
    assert scores.shape == (2,)


def test_tune_C(monkeypatch):
    def fake_cv_f1(_K, _y, _splits, C):
        return np.array([0.1 + C, 0.2 + C])

    monkeypatch.setattr(q, "cv_f1", fake_cv_f1)
    K = np.eye(4)
    y = np.array([0, 1, 0, 1])
    best_C, best_score, per_c = q.tune_C(K, y, splits=[])

    assert best_C == max(q.C_GRID)
    assert best_score == per_c[best_C]


def test_tune_rbf(monkeypatch):
    def fake_tune_C(K_train, _y, _splits):
        score = float(np.mean(K_train))
        return 1.0, score, {1.0: score}

    monkeypatch.setattr(q, "tune_C", fake_tune_C)
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 1, 0, 1])
    best = q.tune_rbf(X, y, train_idx=np.array([0, 1, 2]), splits=[], gammas=(0.01, 1.0))

    assert set(["gamma", "C", "cv_f1", "K"]).issubset(best.keys())


def test_select_quantum_kernel(monkeypatch):
    monkeypatch.setattr(q, "FEATURE_MAPS", ("Z",))
    monkeypatch.setattr(q, "REPS_GRID", (1,))
    monkeypatch.setattr(q, "BANDWIDTHS", (0.1, 0.2))
    monkeypatch.setattr(q, "C_GRID", (0.1, 1.0))

    def fake_quantum_gram(_fm, _nq, _reps, X):
        marker = float(np.mean(X))
        return np.full((4, 4), marker)

    def fake_tune_C(K_train, _y, _splits):
        marker = float(np.mean(K_train))
        per_c = {0.1: marker, 1.0: marker + 0.01}
        return 1.0, per_c[1.0], per_c

    monkeypatch.setattr(q, "quantum_gram", fake_quantum_gram)
    monkeypatch.setattr(q, "tune_C", fake_tune_C)

    Z_all = np.ones((4, 2))
    y_train = np.array([0, 1, 0, 1])
    tr = np.array([0, 1, 2, 3])
    grid, best = q.select_quantum_kernel(Z_all, y_train, tr, splits=[])

    assert ("Z", 1) in grid
    assert best["bw"] == 0.2
    assert best["C"] == 1.0


def test_fit_qsvc(monkeypatch):
    class DummyKernel:
        def __init__(self, feature_map):
            self.feature_map = feature_map

    class DummyQSVC:
        def __init__(self, quantum_kernel, C, class_weight):
            self.quantum_kernel = quantum_kernel
            self.C = C
            self.class_weight = class_weight
            self.support_ = [0, 1]

        def fit(self, X, y):
            self._n = len(X)
            self._y = y

        def predict(self, X):
            return np.array([0] * len(X))

        def decision_function(self, X):
            return np.full(len(X), -0.5)

    monkeypatch.setattr(q, "FidelityStatevectorKernel", DummyKernel)
    monkeypatch.setattr(q, "QSVC", DummyQSVC)
    monkeypatch.setattr(q, "build_feature_map", lambda *_: "fm")

    cfg = {"fm": "Z", "reps": 1, "C": 1.0, "bw": 0.5}
    Z_all = np.ones((6, 3))
    y_train = np.array([0, 1, 0, 1])
    pred, score, elapsed, n_sv = q.fit_qsvc(cfg, Z_all, y_train, tr=np.array([0, 1, 2, 3]), te=np.array([4, 5]))

    assert pred.tolist() == [0, 0]
    assert np.allclose(score, np.array([0.5, 0.5]))
    assert elapsed >= 0.0
    assert n_sv == 2


def test_compute_metrics_and_format_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 1, 1])
    score = np.array([0.8, 0.7, 0.2, 0.1])

    m = q.compute_metrics(y_true, y_pred, score)
    txt = q.format_metrics(m)

    assert m["precision_malignant"] == 0.0
    assert "acc=" in txt and "AUC=" in txt


def test_centered_alignment_regular_and_zero_norm():
    K = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([0, 1])
    val = q.centered_alignment(K, y)
    assert np.isfinite(val)

    K_zero = np.ones((2, 2))
    val_zero = q.centered_alignment(K_zero, y)
    assert val_zero == 0.0


def test_corrected_paired_ttest_edge_and_regular():
    t, p = q.corrected_paired_ttest(np.array([0.1]), n_train=10, n_test=2)
    assert (t, p) == (0.0, 1.0)

    t, p = q.corrected_paired_ttest(np.array([0.2, 0.1, -0.1, 0.05]), n_train=20, n_test=5)
    assert np.isfinite(t)
    assert 0.0 <= p <= 1.0


def test_set_style_changes_rcparams():
    q.set_style()
    assert plt.rcParams["axes.titleweight"] == "bold"


def test_save_and_entity_color(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "OUT_DIR", tmp_path)
    fig, _ = plt.subplots()
    q.save(fig, "t.png")
    assert (tmp_path / "t.png").exists()

    assert q.entity_color("QSVC demo") == q.QUANTUM
    assert q.entity_color("SVC RBF same") == q.CLASSICAL
    assert q.entity_color("other") == q.INK_3


def test_plot_circuit_success_and_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "OUT_DIR", tmp_path)

    class OkCircuit:
        def decompose(self):
            return self

        def draw(self, output="mpl", **_kwargs):
            if output == "mpl":
                fig, _ = plt.subplots()
                return fig
            return "text"

        def depth(self):
            return 3

        def num_nonlocal_gates(self):
            return 2

    monkeypatch.setattr(q, "build_feature_map", lambda *_args, **_kwargs: OkCircuit())
    q.plot_circuit("Z", 1, "label", "ok_plot")
    assert (tmp_path / "ok_plot.png").exists()

    class FallbackCircuit(OkCircuit):
        def draw(self, output="mpl", **_kwargs):
            if output == "mpl":
                raise RuntimeError("no mpl")
            return "text-circuit"

    monkeypatch.setattr(q, "build_feature_map", lambda *_args, **_kwargs: FallbackCircuit())
    q.plot_circuit("Z", 1, "label", "fallback_plot")
    assert (tmp_path / "fallback_plot.txt").exists()


def test_plot_helpers_generate_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "OUT_DIR", tmp_path)

    # Keep grids tiny for a fast, deterministic plotting test.
    monkeypatch.setattr(q, "FEATURE_MAPS", ("Z",))
    monkeypatch.setattr(q, "REPS_GRID", (1,))
    monkeypatch.setattr(q, "BANDWIDTHS", (0.1, 0.2))
    monkeypatch.setattr(q, "C_GRID", (1.0, 2.0))

    grid = {
        ("Z", 1): {
            (0.1, 1.0): 0.5,
            (0.1, 2.0): 0.6,
            (0.2, 1.0): 0.7,
            (0.2, 2.0): 0.8,
        }
    }
    best = {"fm": "Z", "reps": 1, "bw": 0.2, "C": 2.0}
    q.plot_model_selection(grid, best)
    assert (tmp_path / "02_qsvc_model_selection.png").exists()

    test_results = {
        "SVC RBF tuned": {"accuracy": 0.8, "f1_malignant": 0.7, "recall_malignant": 0.6, "roc_auc": 0.85},
        "QSVC Z": {"accuracy": 0.82, "f1_malignant": 0.72, "recall_malignant": 0.62, "roc_auc": 0.87},
    }
    q.plot_leaderboard(test_results, n_test=10)
    assert (tmp_path / "04_test_leaderboard.png").exists()

    y_test = np.array([0, 0, 1, 1])
    curves = {"SVC RBF tuned": np.array([0.9, 0.8, 0.3, 0.2]), "QSVC Z": np.array([0.95, 0.85, 0.25, 0.15])}
    preds = {"QSVC Z": np.array([0, 0, 1, 1]), "SVC RBF tuned": np.array([0, 1, 1, 1])}
    q.plot_roc_confusion(y_test, curves, preds)
    assert (tmp_path / "05_roc_confusion.png").exists()

    diffs = {
        "QSVC vs SVC": {
            "diffs": np.array([0.1, -0.05, 0.02]),
            "p": 0.5,
        }
    }
    q.plot_significance(diffs)
    assert (tmp_path / "03_paired_significance.png").exists()


def test_pipeline_helpers(monkeypatch):
    split = q.SplitArtifacts(
        X_train=np.ones((4, 3)),
        X_test=np.ones((2, 3)),
        y_train=np.array([0, 1, 0, 1]),
        y_test=np.array([0, 1]),
        X_all=np.ones((6, 3)),
        y_all=np.array([0, 1, 0, 1, 0, 1]),
        train_idx=np.array([0, 1, 2, 3]),
        test_idx=np.array([4, 5]),
        cv_splits=[(np.array([0, 1]), np.array([2, 3]))],
    )

    monkeypatch.setattr(q, "tune_rbf", lambda *_args, **_kwargs: {"K": np.eye(6), "C": 1.0, "cv_f1": 0.5, "gamma": 0.1})
    monkeypatch.setattr(q, "fit_eval_precomputed", lambda *_args, **_kwargs: (np.array([0, 1]), np.array([0.1, 0.2])))
    best, out = q.run_classical_baseline_all_features(split)
    assert best["C"] == 1.0
    assert len(out[0]) == 2

    monkeypatch.setattr(q, "apply_superposition_encoding", lambda *_args, **_kwargs: np.ones((6, 2)))
    monkeypatch.setattr(q, "select_quantum_kernel", lambda *_args, **_kwargs: ({"g": 1}, {"fm": "Z", "reps": 1, "bw": 0.1, "C": 1.0, "K": np.eye(6)}))
    monkeypatch.setattr(q, "fit_qsvc", lambda *_args, **_kwargs: (np.array([0, 1]), np.array([0.2, 0.3]), 0.1, 2))
    encoded, grid, best_q, q_out = q.run_quantum_pipeline(split)
    assert encoded.shape == (6, 2)
    assert best_q["fm"] == "Z"
    assert len(q_out[0]) == 2

    best_same, same_out = q.run_classical_baseline_same_encoded_features(split, np.ones((6, 2)))
    assert best_same["gamma"] == 0.1
    assert len(same_out[0]) == 2


def test_main_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "OUT_DIR", tmp_path)

    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    X = np.arange(32.0).reshape(8, 4)

    split = q.SplitArtifacts(
        X_train=X[:6],
        X_test=X[6:],
        y_train=y[:6],
        y_test=y[6:],
        X_all=X,
        y_all=y,
        train_idx=np.array([0, 1, 2, 3, 4, 5]),
        test_idx=np.array([6, 7]),
        cv_splits=[(np.array([0, 1, 2, 3]), np.array([4, 5]))],
    )

    monkeypatch.setattr(q, "load_breast_cancer_data", lambda: (X, y))
    monkeypatch.setattr(q, "prepare_test_and_validation_data", lambda _X, _y: split)

    monkeypatch.setattr(
        q,
        "run_classical_baseline_all_features",
        lambda _split: (
            {"cv_f1": 0.5, "gamma": 0.1, "C": 1.0, "K": np.eye(8)},
            (np.array([0, 1]), np.array([0.8, 0.2])),
        ),
    )
    monkeypatch.setattr(
        q,
        "run_quantum_pipeline",
        lambda _split: (
            np.ones((8, 2)),
            {("Z", 1): {(0.1, 1.0): 0.6}},
            {"fm": "Z", "reps": 1, "bw": 0.1, "C": 1.0, "cv_f1": 0.6, "K": np.eye(8)},
            (np.array([0, 1]), np.array([0.85, 0.15]), 0.01, 2),
        ),
    )
    monkeypatch.setattr(
        q,
        "run_classical_baseline_same_encoded_features",
        lambda _split, _enc: (
            {"cv_f1": 0.55, "gamma": 0.2, "C": 1.0, "K": np.eye(8)},
            (np.array([0, 1]), np.array([0.82, 0.18])),
        ),
    )

    class DummyRSKF:
        def __init__(self, **_kwargs):
            pass

        def split(self, X_train, y_train):
            _ = X_train, y_train
            return [(np.array([0, 1, 2, 3]), np.array([4, 5]))]

    monkeypatch.setattr(q, "RepeatedStratifiedKFold", DummyRSKF)
    monkeypatch.setattr(q, "cv_f1", lambda *_args, **_kwargs: np.array([0.6]))
    monkeypatch.setattr(q, "corrected_paired_ttest", lambda *_args, **_kwargs: (0.0, 1.0))
    monkeypatch.setattr(q, "centered_alignment", lambda *_args, **_kwargs: 0.1)

    monkeypatch.setattr(q, "plot_circuit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(q, "plot_model_selection", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(q, "plot_significance", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(q, "plot_leaderboard", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(q, "plot_roc_confusion", lambda *_args, **_kwargs: None)

    q.main()

    assert (tmp_path / "results.json").exists()
