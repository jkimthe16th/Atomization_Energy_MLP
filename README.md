# Multi-layer Perceptrons for Molecular Atomization Energies

**Author:** D.H. (John) Kim  
**Course:** SPC707P Deep Learning  
**Date:** 15/04/2026

## Abstract

This repository contains a rigorous study and implementation of molecular atomization energy prediction using Coulomb-matrix-derived descriptors and multi-layer perceptrons (MLPs). Working with 57,462 equilibrium organic molecules from the ANI-1E dataset ($\omega$B97x/6-31G(d) level of theory), we develop and evaluate a progression of molecular representations—Coulomb matrix eigenvalues, sorted Coulomb matrices, and the Bag of Bonds descriptor. These are grounded in the nuclear Coulomb interaction operator from the molecular Hamiltonian. 

Our best model, a 5-member ensemble of MLPs operating on concatenated Bag of Bonds and eigenvalue features (564 dimensions), achieves a test MAE of 3.36 kcal/mol (0.15 eV, $R^2 = 0.9996$) on an unseen test set of 8,620 molecules.

## Theory & Fundamentals

Predicting molecular energies from atomic coordinates requires solving the time-independent Schrödinger equation:

$$\hat{H} \Psi = E \Psi$$

Solving this exactly is intractable for large systems. This project leverages machine learning to approximate the energy mapping directly from atomic coordinates and nuclear charges.

### Molecular Representation

We utilize the **Coulomb Matrix** ($M$), which encodes nuclear charges ($Z$) and interatomic distances ($R$) into a symmetric matrix:

* **Off-diagonal elements** represent pairwise nuclear repulsions:
    $$M_{ij} = \frac{Z_i Z_j}{|R_i - R_j|}$$
* **Diagonal elements** approximate isolated-atom energies:
    $$M_{ii} = 0.5 Z_i^{2.4}$$

### Feature Engineering

To ensure the neural network receives fixed-size, permutation-invariant inputs, two complementary descriptors are extracted from $M$:
1.  **Eigenvalues:** Eigenvalue decomposition of $M$ yields a sorted vector $v_{\text{eig}}$ invariant to atom permutation.
2.  **Bag of Bonds (BoB):** Groups off-diagonal elements by element pair (e.g., H-C, C-O), sorts each bag in descending order, and pads to a fixed length, producing $v_{\text{BoB}}$.

The final feature vector is the concatenation of both, $x = v_{\text{BoB}} \oplus v_{\text{eig}}$, resulting in 564 dimensions. Features are z-score normalized per dimension.

## Model Architecture

The regression model is a five-layer MLP mapping the 564-dimensional input to a single atomization energy value. 

* **Topology:** 1024 $\rightarrow$ 512 $\rightarrow$ 256 $\rightarrow$ 128 $\rightarrow$ 1
* **Regularization:** Batch Normalization and Dropout (0.15 - 0.20) at each hidden layer.
* **Activation:** GELU
* **Skip Connection:** A linear projection of the raw input is concatenated with the final hidden representation before the output head.

To improve robustness, an **ensemble of 5 independently seeded models** is trained. Predictions are averaged at inference.

## Training & Results

* **Optimizer:** AdamW (initial LR: 1e-4, weight decay: 1e-5)
* **Loss Function:** Huber Loss ($\delta = 0.1$)
* **Scheduling:** ReduceLROnPlateau (factor: 0.3, patience: 15)

The dataset is split 70/15/15 into train (40,223), validation (8,619), and test (8,620) subsets. 

**Test Performance:**
* MAE: 3.36 kcal/mol (0.15 eV, 0.0053 Hartree)
* RMSE: 4.88 kcal/mol
* $R^2$: 0.9996

## Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
