"""
Multi-layer Perceptrons for Molecular Atomization Energies
Author: D.H. (John) Kim
Course: SPC707P Deep Learning
Date:   15/04/2026

Pipeline
--------
1. Download the ANI-1E dataset from Zenodo (~1 GB).
2. Parse XYZ files and extract (elements, coordinates, energy) triples.
3. Compute Coulomb matrices and derive two descriptors:
      - sorted eigenvalue vector  (MAX_ATOMS dimensions)
      - Bag of Bonds (BoB) vector (variable, padded to fixed size)
4. Concatenate descriptors and z-score-normalise per dimension.
5. Split 70 / 15 / 15 into train / validation / test sets.
6. Train an ensemble of N_ENSEMBLE independently-seeded MLPs.
7. Evaluate on the held-out test set and save diagnostic plots.
"""

import os
import glob
import zipfile
from collections import defaultdict

import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Zenodo record for the ANI-1E equilibrium dataset (~1 GB zip of XYZ files).
ZENODO_URL = "https://zenodo.org/record/4081692/files/ani1e.zip"

DATA_DIR = "ani1e"

# Nuclear charges for the elements present in ANI-1E.
ATOMIC_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "S": 16, "Cl": 17}

# Conversion factor: Hartree → kcal/mol.
HARTREE_TO_KCAL = 627.509474

# Maximum number of heavy + hydrogen atoms across all ANI-1 molecules.
MAX_ATOMS = 26

# Sorted element list used to enumerate Bag-of-Bonds pairs.
ELEMENTS = sorted(ATOMIC_NUMBERS.keys())

# All unique unordered element pairs (including homoatomic).
ELEMENT_PAIRS = [
    (e1, e2)
    for i, e1 in enumerate(ELEMENTS)
    for e2 in ELEMENTS[i:]
]

# Training hyper-parameters.
RANDOM_SEED  = 42
N_ENSEMBLE   = 5
BATCH_SIZE   = 256
MAX_EPOCHS   = 300
LR           = 1e-4
WEIGHT_DECAY = 1e-5
HUBER_DELTA  = 0.1

# ---------------------------------------------------------------------------
# 1. Dataset download
# ---------------------------------------------------------------------------

def download_dataset(url: str = ZENODO_URL, dest: str = DATA_DIR) -> None:
    """Download and extract the ANI-1E XYZ dataset from Zenodo if absent."""
    xyz_files = glob.glob(os.path.join(dest, "**", "*.xyz"), recursive=True)
    if os.path.isdir(dest) and xyz_files:
        print(f"[INFO] Dataset found at '{dest}' ({len(xyz_files)} XYZ files). "
              "Skipping download.")
        return

    os.makedirs(dest, exist_ok=True)
    zip_path = os.path.join(dest, "_download.zip")

    print(f"[INFO] Downloading ANI-1E from:\n       {url}")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(zip_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=65536):
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(
                        f"\r  {pct:5.1f}%  "
                        f"({downloaded / 1e6:.1f} / {total / 1e6:.1f} MB)",
                        end="",
                        flush=True,
                    )
    print()

    print("[INFO] Extracting archive …")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    os.remove(zip_path)
    print("[INFO] Dataset ready.")


# ---------------------------------------------------------------------------
# 2. XYZ parsing
# ---------------------------------------------------------------------------

def _parse_xyz_file(path: str) -> list:
    """
    Parse a (possibly multi-frame) XYZ file.

    Returns a list of ``(elements, coords, energy)`` tuples where:
      - ``elements`` is a list of element symbols,
      - ``coords``   is a float64 array of shape (n_atoms, 3) in Ångströms,
      - ``energy``   is the DFT energy in Hartree (float).
    """
    molecules = []
    with open(path, "r") as fh:
        lines = fh.readlines()

    i = 0
    while i < len(lines):
        # --- atom count ---
        try:
            n_atoms = int(lines[i].strip())
        except (ValueError, IndexError):
            i += 1
            continue

        if i + 1 + n_atoms >= len(lines):
            break

        # --- comment line: extract first float as energy ---
        comment = lines[i + 1].strip()
        energy = None
        for token in comment.replace("=", " ").split():
            try:
                energy = float(token)
                break
            except ValueError:
                continue

        if energy is None:
            i += 2 + n_atoms
            continue

        # --- atom lines ---
        elems = []
        coords = []
        for j in range(i + 2, i + 2 + n_atoms):
            parts = lines[j].split()
            sym = parts[0]
            if sym not in ATOMIC_NUMBERS:
                break
            elems.append(sym)
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

        if len(elems) == n_atoms:
            molecules.append((elems, np.array(coords, dtype=np.float64), energy))

        i += 2 + n_atoms

    return molecules


def load_dataset(data_dir: str = DATA_DIR) -> list:
    """Load all XYZ files under *data_dir* and return a flat molecule list."""
    xyz_paths = sorted(
        glob.glob(os.path.join(data_dir, "**", "*.xyz"), recursive=True)
    )
    if not xyz_paths:
        raise FileNotFoundError(
            f"No XYZ files found under '{data_dir}'. "
            "Run download_dataset() first."
        )

    print(f"[INFO] Parsing {len(xyz_paths)} XYZ file(s) …")
    molecules = []
    for path in xyz_paths:
        molecules.extend(_parse_xyz_file(path))

    print(f"[INFO] Loaded {len(molecules):,} molecules.")
    return molecules


# ---------------------------------------------------------------------------
# 3. Coulomb matrix
# ---------------------------------------------------------------------------

def coulomb_matrix(elements: list, coords: np.ndarray) -> np.ndarray:
    """
    Compute the Coulomb matrix for a single molecule.

    Diagonal:     M_ii = 0.5 * Z_i^2.4
    Off-diagonal: M_ij = Z_i * Z_j / |R_i - R_j|
    """
    n = len(elements)
    Z = np.array([ATOMIC_NUMBERS[e] for e in elements], dtype=np.float64)
    M = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        M[i, i] = 0.5 * Z[i] ** 2.4
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            if d > 1e-8:
                M[i, j] = M[j, i] = Z[i] * Z[j] / d

    return M


# ---------------------------------------------------------------------------
# 4. Feature engineering
# ---------------------------------------------------------------------------

def eigenvalue_features(M: np.ndarray, max_atoms: int = MAX_ATOMS) -> np.ndarray:
    """
    Return the sorted (descending, by magnitude) eigenvalue vector of *M*,
    zero-padded to *max_atoms*.
    """
    eigvals = np.linalg.eigvalsh(M)
    eigvals = np.sort(np.abs(eigvals))[::-1]
    vec = np.zeros(max_atoms, dtype=np.float64)
    n = min(len(eigvals), max_atoms)
    vec[:n] = eigvals[:n]
    return vec


def _bag_of_bonds_sizes(molecules: list) -> dict:
    """
    Determine the maximum number of bonds for each element pair across the
    full dataset. Returns an ordered dict ``{(e1, e2): max_count}``.
    """
    pair_counts: dict = {p: 0 for p in ELEMENT_PAIRS}
    for elems, _, _ in molecules:
        local: dict = defaultdict(int)
        n = len(elems)
        for i in range(n):
            for j in range(i + 1, n):
                pair = tuple(sorted([elems[i], elems[j]]))
                local[pair] += 1
        for pair, cnt in local.items():
            if pair in pair_counts:
                pair_counts[pair] = max(pair_counts[pair], cnt)
    return pair_counts


def _bag_of_bonds_features(
    elements: list,
    coords: np.ndarray,
    pair_sizes: dict,
) -> np.ndarray:
    """
    Compute the Bag-of-Bonds descriptor for one molecule.

    For each element pair the Coulomb interaction values Z_i*Z_j/r_ij are
    collected, sorted descending, and zero-padded to *pair_sizes[pair]*.
    """
    bags: dict = {p: [] for p in pair_sizes}
    Z = np.array([ATOMIC_NUMBERS[e] for e in elements], dtype=np.float64)
    n = len(elements)

    for i in range(n):
        for j in range(i + 1, n):
            pair = tuple(sorted([elements[i], elements[j]]))
            if pair in bags:
                d = np.linalg.norm(coords[i] - coords[j])
                if d > 1e-8:
                    bags[pair].append(Z[i] * Z[j] / d)

    vec = []
    for p in pair_sizes:
        vals = sorted(bags[p], reverse=True)
        size = pair_sizes[p]
        padded = vals + [0.0] * (size - len(vals))
        vec.extend(padded[:size])

    return np.array(vec, dtype=np.float64)


def build_features(molecules: list) -> tuple:
    """
    Build the full feature matrix *X* and target vector *y*.

    Each row of *X* is the concatenation of the BoB descriptor and the
    eigenvalue descriptor. *y* contains atomization energies in kcal/mol.
    """
    print("[INFO] Computing Bag-of-Bonds vocabulary …")
    pair_sizes = _bag_of_bonds_sizes(molecules)
    bob_dim = sum(pair_sizes.values())
    eig_dim = MAX_ATOMS
    feat_dim = bob_dim + eig_dim
    print(
        f"[INFO] BoB dim = {bob_dim}, Eig dim = {eig_dim}, "
        f"Total = {feat_dim}"
    )

    n = len(molecules)
    X = np.zeros((n, feat_dim), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)

    for idx, (elems, coords, energy) in enumerate(molecules):
        if idx % 5000 == 0:
            print(f"  {idx:6d} / {n}")

        M = coulomb_matrix(elems, coords)
        ev = eigenvalue_features(M)
        bb = _bag_of_bonds_features(elems, coords, pair_sizes)

        X[idx] = np.concatenate([bb, ev]).astype(np.float32)
        y[idx] = float(energy) * HARTREE_TO_KCAL

    return X, y


# ---------------------------------------------------------------------------
# 5. Model definition
# ---------------------------------------------------------------------------

def build_model(input_dim: int, seed: int = 0) -> keras.Model:
    """
    Build a single MLP with the architecture described in the README:
      1024 → 512 → 256 → 128 → 1
    with BatchNorm, GELU activation, Dropout at each hidden layer, and a
    linear skip connection from the raw input to the penultimate layer.
    """
    tf.random.set_seed(seed)

    inp = keras.Input(shape=(input_dim,))

    # Hidden layers
    x = layers.Dense(1024)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)
    x = layers.Dropout(0.20)(x)

    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)
    x = layers.Dropout(0.20)(x)

    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)
    x = layers.Dropout(0.15)(x)

    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)
    x = layers.Dropout(0.15)(x)

    # Linear skip connection: project raw input to 128 dims and concatenate.
    skip = layers.Dense(128, use_bias=False)(inp)
    x = layers.Concatenate()([x, skip])

    # Output head
    out = layers.Dense(1)(x)

    return keras.Model(inp, out)


def _compile_model(model: keras.Model) -> keras.Model:
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=LR, weight_decay=WEIGHT_DECAY
    )
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(delta=HUBER_DELTA),
        metrics=["mae"],
    )
    return model


# ---------------------------------------------------------------------------
# 6. Training
# ---------------------------------------------------------------------------

def train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n: int = N_ENSEMBLE,
) -> tuple:
    """Train *n* independently-seeded MLP models and return them with their
    training histories."""
    models = []
    histories = []

    for i in range(n):
        print(f"\n[INFO] Training model {i + 1} / {n} …")
        tf.random.set_seed(RANDOM_SEED + i)
        model = build_model(X_train.shape[1], seed=RANDOM_SEED + i)
        _compile_model(model)

        callbacks = [
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.3,
                patience=15,
                min_lr=1e-7,
                verbose=1,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=40,
                restore_best_weights=True,
                verbose=1,
            ),
        ]

        hist = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
        )
        models.append(model)
        histories.append(hist)

    return models, histories


def ensemble_predict(models: list, X: np.ndarray) -> np.ndarray:
    """Average predictions from all ensemble members."""
    preds = np.stack(
        [m.predict(X, verbose=0).ravel() for m in models], axis=0
    )
    return preds.mean(axis=0)


# ---------------------------------------------------------------------------
# 7. Evaluation & plots
# ---------------------------------------------------------------------------

def evaluate(
    y_true: np.ndarray, y_pred: np.ndarray, label: str = "Test"
) -> tuple:
    """Print and return MAE, RMSE, R²."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - float(np.mean(y_true))) ** 2)
    r2 = float(1.0 - ss_res / ss_tot)
    print(
        f"\n[{label}]  MAE = {mae:.2f} kcal/mol | "
        f"RMSE = {rmse:.2f} kcal/mol | R² = {r2:.4f}"
    )
    return mae, rmse, r2


def plot_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mae: float,
    rmse: float,
    r2: float,
    histories: list,
) -> None:
    """Save three diagnostic figures to the working directory."""

    # ── Parity plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=2, alpha=0.3, color="steelblue", rasterized=True)
    lo = min(float(y_true.min()), float(y_pred.min()))
    hi = max(float(y_true.max()), float(y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("True Atomization Energy (kcal/mol)")
    ax.set_ylabel("Predicted Atomization Energy (kcal/mol)")
    ax.set_title(
        f"Ensemble Parity Plot\n"
        f"MAE = {mae:.2f}  RMSE = {rmse:.2f}  R² = {r2:.4f}"
    )
    plt.tight_layout()
    plt.savefig("parity_plot.png", dpi=150)
    plt.close(fig)
    print("[INFO] Saved parity_plot.png")

    # ── Residual distribution ────────────────────────────────────────────────
    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(0, color="red", lw=1.5, ls="--")
    ax.set_xlabel("Residual (kcal/mol)")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")
    plt.tight_layout()
    plt.savefig("residual_distribution.png", dpi=150)
    plt.close(fig)
    print("[INFO] Saved residual_distribution.png")

    # ── Per-model learning curves ────────────────────────────────────────────
    n_models = len(histories)
    fig, axes = plt.subplots(
        1, n_models, figsize=(4 * n_models, 4), sharey=True
    )
    if n_models == 1:
        axes = [axes]
    for i, (hist, ax_) in enumerate(zip(histories, axes)):
        ax_.plot(hist.history["loss"], label="Train")
        ax_.plot(hist.history["val_loss"], label="Val")
        ax_.set_xlabel("Epoch")
        ax_.set_title(f"Model {i + 1}")
        ax_.legend(fontsize=7)
    axes[0].set_ylabel("Huber Loss")
    plt.suptitle("Learning Curves")
    plt.tight_layout()
    plt.savefig("learning_curves.png", dpi=150)
    plt.close(fig)
    print("[INFO] Saved learning_curves.png")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # 1. Download dataset
    download_dataset()

    # 2. Load & parse XYZ files
    molecules = load_dataset()

    # 3. Build feature matrix
    X, y = build_features(molecules)
    print(f"[INFO] Feature matrix: {X.shape},  targets: {y.shape}")

    # 4. Z-score normalise features
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    # 5. 70 / 15 / 15 split
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=RANDOM_SEED
    )
    print(
        f"[INFO] Split — Train: {len(y_train):,}  "
        f"Val: {len(y_val):,}  Test: {len(y_test):,}"
    )

    # 6. Train ensemble
    models, histories = train_ensemble(X_train, y_train, X_val, y_val)

    # 7. Evaluate on held-out test set
    y_pred = ensemble_predict(models, X_test)
    mae, rmse, r2 = evaluate(y_test, y_pred, "Test")

    # 8. Save diagnostic plots
    plot_results(y_test, y_pred, mae, rmse, r2, histories)

    print("\n[INFO] Pipeline complete.")


if __name__ == "__main__":
    main()
