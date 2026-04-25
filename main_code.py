#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#-------------------------------------------------------------------------
# Multi-layer Perceptrons for Molecular Atomization Energies
#-------------------------------------------------------------------------

### Cell 1: Imports & Configuration 

import os, io, re, sys, time, gzip, tarfile, zipfile, json
from pathlib import Path
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print(f"TensorFlow {tf.__version__}")
print(f"GPU available: {bool(tf.config.list_physical_devices('GPU'))}")

WORK_DIR = Path("/home/jovyan/deep_learning_2026/Final Project/")
WORK_DIR.mkdir(parents=True, exist_ok=True)

ANI1E_RECORD = "4680953"
ANI1E_DIR = WORK_DIR / "ani1e"
ANI1E_DIR.mkdir(exist_ok=True)

ELEMENTS = [1, 6, 7, 8]
ELEM_NAMES = {1: "H", 6: "C", 7: "N", 8: "O"}
ELEMENT_TO_Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "I": 53, "Xe": 54,
}

FREE_ATOM_ENERGY = {1: -0.500273, 6: -37.846772, 7: -54.583861, 8: -75.064579}

PAIRS = list(combinations_with_replacement(ELEMENTS, 2))

N_ENSEMBLE = 5
BATCH_SIZE = 512
MAX_EPOCHS = 500
PATIENCE = 40

HARTREE_TO_KCAL = 627.509
HARTREE_TO_EV = 27.2114


### Cell 2: Data Acquisition 

def download_with_progress(url: str, dest: Path, chunk_mb: int = 4) -> None:
    chunk = chunk_mb * (1 << 20)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        t0 = time.time()
        with open(dest, "wb") as fh:
            for data in r.iter_content(chunk_size=chunk):
                fh.write(data)
                downloaded += len(data)
                if total:
                    pct = downloaded / total * 100
                    speed = downloaded / (time.time() - t0 or 0.001) / 1e6
                    print(
                        f"\r  {pct:5.1f}%  {downloaded / 1e9:.2f}/{total / 1e9:.2f} GB"
                        f"  {speed:.1f} MB/s",
                        end="", flush=True,)
    print()

def fetch_ani1e_files() -> None:
    api_url = f"https://zenodo.org/api/records/{ANI1E_RECORD}"
    record = requests.get(api_url, timeout=30).json()
    for f_info in record["files"]:
        fname = f_info["key"]
        local = ANI1E_DIR / fname
        if not local.exists():
            download_with_progress(f_info["links"]["self"], local)

fetch_ani1e_files()

for f in sorted(ANI1E_DIR.iterdir()):
    size_mb = f.stat().st_size / 1e6
    print(f"  {f.name:40s}  {size_mb:8.2f} MB")


### Cell 3: xyz parsing and energy extraction 

def parse_concatenated_xyz(text: str) -> list[dict]:
    lines = text.strip().split("\n")
    molecules = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        try:
            n_atoms = int(line)
        except ValueError:
            i += 1
            continue
        if i + 1 + n_atoms > len(lines):
            break

        comment = lines[i + 1].strip()
        atoms_z, coords, valid = [], [], True

        for j in range(n_atoms):
            parts = lines[i + 2 + j].split()
            if len(parts) < 4:
                valid = False
                break
            elem = parts[0].strip().capitalize()
            z = ELEMENT_TO_Z.get(elem)
            if z is None:
                valid = False
                break
            try:
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
                atoms_z.append(z)
            except ValueError:
                valid = False
                break

        if valid and atoms_z:
            molecules.append({
                "Z": atoms_z,
                "coords": np.array(coords),
                "n_atoms": n_atoms,
                "comment": comment,})
        i += 2 + n_atoms
    return molecules


def extract_energy_from_xyz_comment(comment: str) -> float | None:
    match = re.search(r"(?:energy|E|E0|total_energy)\s*[=:]\s*([-\d.eE+]+)",comment, re.IGNORECASE,)
    if match:
        try:
            v = float(match.group(1))
            if v < 0:
                return v
        except ValueError:
            pass

    floats = re.findall(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", comment)
    candidates = [float(f) for f in floats if float(f) < -5]
    if candidates:
        return min(candidates)

    if len(floats) == 1:
        try:
            return float(floats[0])
        except ValueError:
            pass
    return None


def _process_xyz_text(text: str, dest: list) -> None:
    for mol in parse_concatenated_xyz(text):
        e = extract_energy_from_xyz_comment(mol["comment"])
        if e is not None:
            mol["energy_hartree"] = e
            dest.append(mol)


def load_ani1e_molecules() -> list[dict]:
    molecules = []
    for fpath in sorted(ANI1E_DIR.iterdir()):
        text = None
        if fpath.name.endswith(".tar.gz") or fpath.name.endswith(".tgz"):
            try:
                with tarfile.open(fpath, "r:gz") as tf_file:
                    for member in tf_file:
                        if member.name.endswith(".xyz") and not member.isdir():
                            ef = tf_file.extractfile(member)
                            if ef:
                                _process_xyz_text(ef.read().decode("utf-8", errors="replace"),molecules,)
            except Exception:
                continue
        elif fpath.name.endswith(".zip"):
            try:
                with zipfile.ZipFile(fpath) as zf:
                    for name in zf.namelist():
                        if name.endswith(".xyz"):
                            _process_xyz_text(zf.read(name).decode("utf-8", errors="replace"),molecules,)
            except Exception:
                continue
        elif fpath.name.endswith(".xyz.gz") or fpath.name.endswith(".xyz_Ener.gz"):
            try:
                with gzip.open(fpath, "rt", encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except Exception:
                continue
        elif fpath.suffix == ".xyz":
            text = fpath.read_text(encoding="utf-8", errors="replace")
        else:
            continue

        if text:
            _process_xyz_text(text, molecules)

    return molecules


def compute_atomization_energies(molecules: list[dict]) -> list[dict]:
    for mol in molecules:
        mol["atomization_energy"] = mol["energy_hartree"] - sum(FREE_ATOM_ENERGY[z] for z in mol["Z"])
    return [m for m in molecules if m.get("atomization_energy") is not None]


molecules = load_ani1e_molecules()
molecules = compute_atomization_energies(molecules)
print(f"Loaded {len(molecules)} molecules with valid atomization energies")

Z_to_sym = {1: "H", 6: "C", 7: "N", 8: "O"}

for idx in [0, 1, 2]:
    mol = molecules[idx]
    formula = "".join(Z_to_sym.get(z, "?") for z in mol["Z"])
    print(f"--- Molecule {idx} : {formula}  ({mol['n_atoms']} atoms) ---")
    print(f"  Total energy   : {mol['energy_hartree']:.6f} Ha")
    print(f"  Atomization E  : {mol['atomization_energy']:.6f} Ha"
          f"  ({mol['atomization_energy'] * HARTREE_TO_KCAL:.2f} kcal/mol)")
    print(f"  Coordinates (first 4 atoms):")
    for k in range(min(4, mol["n_atoms"])):
        sym = Z_to_sym.get(mol["Z"][k], "?")
        x, y, z = mol["coords"][k]
        print(f"    {sym:2s}  {x:10.6f}  {y:10.6f}  {z:10.6f}")
    print()

# Dataset overview
n_atoms_list = [m["n_atoms"] for m in molecules]
ae_list = [m["atomization_energy"] * HARTREE_TO_KCAL for m in molecules]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(n_atoms_list, bins=range(1, max(n_atoms_list) + 2), edgecolor="black", alpha=0.7, color="steelblue")
axes[0].set(xlabel="Number of atoms", ylabel="Count", title="Molecule size distribution")
axes[0].grid(alpha=0.3)

axes[1].hist(ae_list, bins=80, edgecolor="black", alpha=0.7, color="coral")
axes[1].set(xlabel="Atomization energy (kcal/mol)", ylabel="Count", title="Target distribution")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()


### Cell 4: Feature engineergin 

def determine_bob_dimensions(molecules: list[dict]) -> dict[tuple, int]:
    max_counts = {pair: 0 for pair in PAIRS}
    for mol in molecules:
        Z = mol["Z"]
        n = len(Z)
        counts = {pair: 0 for pair in PAIRS}
        for i in range(n):
            counts[(Z[i], Z[i])] = counts.get((Z[i], Z[i]), 0) + 1
        for i in range(n):
            for j in range(i + 1, n):
                key = (min(Z[i], Z[j]), max(Z[i], Z[j]))
                counts[key] = counts.get(key, 0) + 1
        for pair in PAIRS:
            max_counts[pair] = max(max_counts[pair], counts.get(pair, 0))
    return max_counts


def compute_features(
    Z: list[int],
    coords: np.ndarray,
    max_counts: dict[tuple, int],
    n_max: int,) -> np.ndarray:
    n = len(Z)
    Z_arr = np.array(Z, dtype=np.float64)
    R = np.array(coords, dtype=np.float64)

    M = np.zeros((n, n))
    np.fill_diagonal(M, 0.5 * Z_arr ** 2.4)
    for i in range(n):
        for j in range(i + 1, n):
            dist = max(np.linalg.norm(R[i] - R[j]), 1e-10)
            M[i, j] = M[j, i] = Z_arr[i] * Z_arr[j] / dist

    pair_values = {pair: [] for pair in PAIRS}
    for i in range(n):
        pair_values[(Z[i], Z[i])].append(M[i, i])
    for i in range(n):
        for j in range(i + 1, n):
            key = (min(Z[i], Z[j]), max(Z[i], Z[j]))
            pair_values[key].append(M[i, j])

    bob = []
    for pair in PAIRS:
        vals = sorted(pair_values.get(pair, []), reverse=True)
        bob.extend(vals + [0.0] * (max_counts[pair] - len(vals)))

    eigs = np.linalg.eigvalsh(M)
    eigs = eigs[np.argsort(-np.abs(eigs))]
    eig_padded = np.zeros(n_max)
    eig_padded[: len(eigs)] = eigs

    return np.concatenate([bob, eig_padded])


def featurise_dataset(molecules: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    max_counts = determine_bob_dimensions(molecules)
    n_max = max(m["n_atoms"] for m in molecules)
    total_dim = sum(max_counts.values()) + n_max

    X = np.zeros((len(molecules), total_dim), dtype=np.float64)
    y = np.zeros(len(molecules), dtype=np.float64)

    for i, mol in enumerate(molecules):
        X[i] = compute_features(mol["Z"], mol["coords"], max_counts, n_max)
        y[i] = mol["atomization_energy"]

    assert np.isfinite(X).all() and np.isfinite(y).all(), "NaN/Inf in features"
    return X, y


X, y = featurise_dataset(molecules)
print(f"Feature matrix: {X.shape}, Target vector: {y.shape}")

# Raw feature vector for one molecule (before any scaling)
print("Sample feature vector (molecule 0), first 30 values:")
print(np.array2string(X[0, :30], precision=4, suppress_small=True))
print(f"...({X.shape[1]} total features)")
print(f"\nNon-zero features per molecule (avg): {np.mean(np.count_nonzero(X, axis=1)):.1f} / {X.shape[1]}")
print(f"Feature range: [{X.min():.4f}, {X.max():.4f}]")
print(f"Target range : [{y.min():.6f}, {y.max():.6f}] Ha")

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_s = scaler_X.fit_transform(X)
y_s = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

fig, axes = plt.subplots(2, 2, figsize=(13, 8))

axes[0, 0].hist(X[:, :30].ravel(), bins=80, color="steelblue", alpha=0.7, edgecolor="black")
axes[0, 0].set(xlabel="Value", ylabel="Count", title="Features BEFORE scaling (first 30 dims)")
axes[0, 0].grid(alpha=0.3)

axes[0, 1].hist(X_s[:, :30].ravel(), bins=80, color="coral", alpha=0.7, edgecolor="black")
axes[0, 1].set(xlabel="Value", ylabel="Count", title="Features AFTER scaling (first 30 dims)")
axes[0, 1].grid(alpha=0.3)

axes[1, 0].hist(y, bins=80, color="steelblue", alpha=0.7, edgecolor="black")
axes[1, 0].set(xlabel="Atomization energy (Ha)", ylabel="Count", title="Targets BEFORE scaling")
axes[1, 0].grid(alpha=0.3)

axes[1, 1].hist(y_s, bins=80, color="coral", alpha=0.7, edgecolor="black")
axes[1, 1].set(xlabel="Scaled value", ylabel="Count", title="Targets AFTER scaling")
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Train / val / test split (70/15/15)
X_train, X_temp, y_train, y_temp = train_test_split(X_s, y_s, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")


### Cell 5: Model definition 

def build_energy_mlp(d_in: int) -> tf.keras.Model:
    """4-block MLP with BatchNorm, GELU, dropout, and a skip connection
    from the input layer to the penultimate representation."""
    inputs = tf.keras.Input(shape=(d_in,))

    x = tf.keras.layers.Dense(1024)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("gelu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("gelu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("gelu")(x)
    x = tf.keras.layers.Dropout(0.15)(x)

    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("gelu")(x)

    skip = tf.keras.layers.Dense(128)(inputs)

    merged = tf.keras.layers.Concatenate()([x, skip])
    output = tf.keras.layers.Dense(1)(merged)

    return tf.keras.Model(inputs=inputs, outputs=output)

tmp_model = build_energy_mlp(X.shape[1])
tmp_model.summary()


### Cell 6: Training 

class ReduceLROnPlateauManual:
    def __init__(self, optimizer, factor=0.3, patience=15, min_lr=1e-6):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best = float("inf")
        self.wait = 0

    def step(self, metric):
        if metric < self.best:
            self.best = metric
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = float(self.optimizer.learning_rate.numpy())
                new_lr = max(old_lr * self.factor, self.min_lr)
                self.optimizer.learning_rate.assign(new_lr)
                self.wait = 0


@tf.function
def train_step(model, optimizer, loss_fn, x_batch, y_batch):
    with tf.GradientTape() as tape:
        predictions = model(x_batch, training=True)
        loss = loss_fn(y_batch, tf.squeeze(predictions, axis=-1))
    gradients = tape.gradient(loss, model.trainable_variables)
    clipped_grads, _ = tf.clip_by_global_norm(gradients, 5.0)
    optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))
    return loss


def train_single_model(X_tr, y_tr, X_vl, y_vl, seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = build_energy_mlp(X_tr.shape[1])
    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-5)
    loss_fn = tf.keras.losses.Huber(delta=0.1)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_tr.astype(np.float32), y_tr.astype(np.float32)))
        .shuffle(len(X_tr), seed=seed)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE))
    X_vl_t = tf.constant(X_vl, dtype=tf.float32)
    y_vl_t = tf.constant(y_vl, dtype=tf.float32)

    scheduler = ReduceLROnPlateauManual(optimizer, factor=0.3, patience=15, min_lr=1e-6)
    optimizer.build(model.trainable_variables)

    best_val, best_epoch, best_weights, wait = float("inf"), 0, None, 0
    train_losses, val_losses = [], []

    print(f"\n>> Training Model (Seed {seed})")

    for epoch in range(MAX_EPOCHS):
        if epoch == 5:
            optimizer.learning_rate.assign(1e-3)

        epoch_loss, n_samples = 0.0, 0
        for xb, yb in train_ds:
            loss = train_step(model, optimizer, loss_fn, xb, yb)
            epoch_loss += loss.numpy() * len(xb)
            n_samples += len(xb)

        val_pred = tf.squeeze(model(X_vl_t, training=False), axis=-1)
        val_mae = tf.reduce_mean(tf.abs(val_pred - y_vl_t)).numpy()

        avg_train_loss = epoch_loss / n_samples
        train_losses.append(avg_train_loss)
        val_losses.append(val_mae)

        if epoch % 20 == 0:
            print(f"   Epoch {epoch:03d} | Train Loss: {avg_train_loss:.6f} | Val MAE: {val_mae:.6f}")

        if epoch >= 5:
            scheduler.step(val_mae)

        if val_mae < best_val:
            best_val, best_epoch, wait = val_mae, epoch, 0
            best_weights = model.get_weights()
        else:
            wait += 1

        if wait >= PATIENCE:
            print(f"   Early stopping at epoch {epoch}. Best Val MAE: {best_val:.6f}")
            break

    model.set_weights(best_weights)
    return model, train_losses, val_losses, best_epoch


def train_ensemble(X_train, y_train, X_val, y_val):
    models, all_tl, all_vl = [], [], []
    for i in range(N_ENSEMBLE):
        seed = 42 + i * 7
        m, tl, vl, _ = train_single_model(X_train, y_train, X_val, y_val, seed)
        models.append(m)
        all_tl.append(tl)
        all_vl.append(vl)
    return models, all_tl, all_vl

models, all_tl, all_vl = train_ensemble(X_train, y_train, X_val, y_val)

models[0].save("MLP_model_Eat.keras")


### Cell 7: Evaluation and diagnostics 

# Per-model training + validation curves side by side
fig, axes = plt.subplots(1, N_ENSEMBLE, figsize=(4 * N_ENSEMBLE, 3.5), sharey=True)
for i in range(N_ENSEMBLE):
    ax = axes[i]
    ax.plot(all_tl[i], label="Train loss", alpha=0.7)
    ax.plot(all_vl[i], label="Val MAE", alpha=0.7)
    ax.set(xlabel="Epoch", title=f"Model {i+1}")
    if i == 0:
        ax.set_ylabel("Loss (scaled)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
plt.suptitle("Per-model training curves", fontsize=13)
plt.tight_layout()
plt.show()

def evaluate_ensemble(models, X_test, y_test, scaler_y, all_val_losses, save_path=None):
    X_test_t = tf.constant(X_test, dtype=tf.float32)

    individual_preds = []
    for m in models:
        pred = tf.squeeze(m(X_test_t, training=False), axis=-1).numpy()
        individual_preds.append(pred)

    ensemble_pred_s = np.mean(individual_preds, axis=0)
    y_pred = scaler_y.inverse_transform(ensemble_pred_s.reshape(-1, 1)).ravel()
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    residuals = y_pred - y_true

    mae_ha = np.mean(np.abs(residuals))
    metrics = {
        "mae_hartree": mae_ha,
        "mae_kcal": mae_ha * HARTREE_TO_KCAL,
        "mae_eV": mae_ha * HARTREE_TO_EV,
        "rmse_hartree": np.sqrt(np.mean(residuals ** 2)),
        "r2": 1 - np.sum(residuals ** 2) / np.sum((y_true - y_true.mean()) ** 2),}

    ind_maes = []
    for ps in individual_preds:
        p = scaler_y.inverse_transform(ps.reshape(-1, 1)).ravel()
        ind_maes.append(np.mean(np.abs(p - y_true)) * HARTREE_TO_KCAL)
    metrics["individual_mae_kcal"] = ind_maes

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (a) Validation loss curves
    ax = axes[0, 0]
    for i, vl in enumerate(all_val_losses):
        ax.plot(vl, alpha=0.5, linewidth=0.7, label=f"Model {i + 1}")
    ax.set(xlabel="Epoch", ylabel="Val MAE (scaled)")
    ax.set_title(f"Training Curves ({N_ENSEMBLE}-Model Ensemble)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) Parity plot
    ax = axes[0, 1]
    ax.scatter(y_true, y_pred, alpha=0.12, s=4, edgecolors="none", c="steelblue")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r-", lw=1.5, label="y = x")
    ax.set(xlabel="True Atomization Energy (Ha)", ylabel="Predicted Atomization Energy (Ha)")
    ax.set_title(f"Parity  |  MAE = {metrics['mae_kcal']:.2f} kcal/mol, R\u00b2 = {metrics['r2']:.4f}")
    ax.legend()
    ax.grid(alpha=0.3)

    # (c) Residuals vs true value
    ax = axes[1, 0]
    ax.scatter(y_true, residuals, alpha=0.12, s=4, edgecolors="none", c="steelblue")
    ax.axhline(0, color="red", lw=1.5)
    ax.set(xlabel="True Atomization Energy (Ha)", ylabel="Residual (Ha)")
    ax.set_title("Residuals")
    ax.grid(alpha=0.3)

    # (d) Error distribution
    res_kcal = residuals * HARTREE_TO_KCAL
    ax = axes[1, 1]
    ax.hist(res_kcal, bins=100, edgecolor="black", alpha=0.7, density=True, color="steelblue")
    ax.axvline(0, color="red", lw=1.5)
    ax.set(xlabel="Residual (kcal/mol)", ylabel="Density")
    ax.set_title(
        f"Error Distribution  "
        f"(\u03bc = {res_kcal.mean():.2f}, \u03c3 = {res_kcal.std():.2f} kcal/mol)")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return metrics

plot_path = WORK_DIR / "rubric_plots_v3.png"
metrics = evaluate_ensemble(models, X_test, y_test, scaler_y, all_vl, save_path=plot_path)

print(f"Test MAE  : {metrics['mae_kcal']:.2f} kcal/mol  "
      f"({metrics['mae_hartree']:.6f} Ha, {metrics['mae_eV']:.4f} eV)")
print(f"Test RMSE : {metrics['rmse_hartree'] * HARTREE_TO_KCAL:.2f} kcal/mol")
print(f"Test R\u00b2   : {metrics['r2']:.6f}")

# Ensemble vs individual model comparison
print("Individual model MAEs (kcal/mol):")
for i, mae in enumerate(metrics["individual_mae_kcal"]):
    print(f"  Model {i+1}: {mae:.2f}")
print(f"  Ensemble: {metrics['mae_kcal']:.2f}")
print(f"  Improvement over best single model: "
      f"{min(metrics['individual_mae_kcal']) - metrics['mae_kcal']:.2f} kcal/mol")

