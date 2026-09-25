## Team Photon_riders


**Team members:**
- Ankit Sharma ([@PhotonCruyo](https://github.com/PhotonCryo))
- Soham ([@soardraspi](https://github.com/soardraspi))

---

## Table of Contents
1. [Introduction](#introduction)
2. [Our approach](#Qur-approach)
3. [Results & Completed deliverables](#Results-&-Completed-deliverables)
   - [Comparison with Classical Solutions](#comparison-with-classical-solutions)
   - [Energy Dynamics](#energy-dynamics)
   - [Parameter Range Observations](#parameter-range-observations)
4. [Key Insights and Future Work](#key-insights-and-future-work)
5. [Conclusion](#conclusion)

## Overview

This project studies the **one-dimensional Axial Next-Nearest-Neighbor Ising (ANNNI) model** and how quantum gate noise affects phase identification.

The model contains three competing effects: ferromagnetic nearest-neighbor coupling, frustrating next-nearest-neighbor coupling, and a transverse magnetic field. By scanning the coupling parameters, we reconstruct the phase structure and then investigate how depolarizing noise alters the observables used to identify the phases.

Our implementation combines **exact diagonalization (ED)** with a **variational quantum eigensolver (VQE)** workflow, allowing the quantum results to be compared against a controlled classical reference.

---

## Problem Statement

We investigate the ANNNI model in the parameter space \((\kappa,h)\):

- **\(\kappa\)** controls the strength of the frustrating next-nearest-neighbor interaction.
- **\(h\)** controls the transverse field.
- **Periodic boundary conditions** are used throughout the main calculations.

The project focuses on two questions:

1. How accurately can the different phases be identified from ground-state observables?
2. How does depolarizing noise on two-qubit gates modify the observed phase structure?

---

## Approach

### 1. Exact Diagonalization

Exact diagonalization provides the reference calculation for the project.

- The ANNNI Hamiltonian is constructed directly in the computational basis.
- Nearest-neighbor and next-nearest-neighbor interactions are implemented with periodic wrap-around.
- The transverse-field term is represented through single-spin flips.
- The Hamiltonian is assembled as a sparse many-body matrix.
- The ground state and corresponding energy are obtained using numerical eigensolvers.
- The main clean scan uses **\(N=8\)** with a **30 × 30 \((\kappa,h)\) grid**.

### 2. Ground-State Observables

The ground state is analyzed using several complementary quantities:

- Nearest-neighbor \(ZZ\) correlations
- Next-nearest-neighbor \(ZZ\) correlations
- Transverse \(X\) magnetization
- Structure factor \(S(q)\)
- Momentum-space peak information

These observables provide signatures for the main phase characters:

- **Ferromagnetic:** strong nearest-neighbor alignment
- **Antiphase:** strong period-four / \(q=\pi/2\) ordering signature
- **Paramagnetic:** stronger response to the transverse field
- **Incommensurate/floating candidate:** finite-\(q\) structure away from the dominant commensurate peaks

Because finite systems can preserve global symmetry, correlation-based observables are used instead of relying only on raw magnetization.

### 3. Phase Classification

A rule-based phase classifier combines the measured observables to assign a phase label at every point of the parameter grid.

The resulting phase map is compared with the analytical Ising, KT, and BKT reference curves supplied with the challenge.

### 4. VQE Cross-Validation

A separate VQE pipeline was implemented as an independent quantum-method check.

- A parameterized ansatz is optimized to minimize the ANNNI energy.
- Physically motivated initializations corresponding to different phase regions were tested.
- Representative VQE ground-state energies were compared against exact diagonalization.
- The same observable-based analysis can then be applied to the variational states.

This provides an independent route to the phase diagram rather than relying on a single computational method.

### 5. Depolarizing Noise

The noisy simulation follows the challenge convention:

- A depolarizing channel is applied after each CNOT.
- Noise levels studied are \(p=0\), \(0.01\), and \(0.05\).
- Noisy circuits are simulated with a mixed-state backend.
- The effect of noise is tracked through the same physical observables used in the clean analysis.

This makes it possible to examine how correlations and phase signatures degrade as gate noise increases.

---

## Scalability and Simulation Optimization

To reduce the cost of repeated quantum-circuit simulations, we also explored a higher-performance simulation workflow.

- Circuit execution and optimization were migrated to **TensorCircuit-ng**.
- The **JAX backend** with **Optax** was used for accelerated optimization.
- Simulation precision can be adjusted to trade numerical accuracy for execution speed.
- The internal QR-decomposition path was adapted for **Apple M2 Metal GPU** execution.
- For a representative circuit run, the optimized workflow reduced execution time from roughly **90 seconds to about 2 seconds**.
- The same approach provides a route toward larger system sizes, with reduced precision used when necessary for local computation.
- Further circuit-contraction optimization can be explored with methods available through **cotengra**, including Greedy and KaHyPar strategies.

---

## Validation Strategy

The implementation is checked at multiple levels:

1. **Local physical validation**  
   Representative parameter points are checked for the expected phase signatures.

2. **ED–VQE comparison**  
   VQE energies are compared directly with exact ground-state energies.

3. **Phase-diagram validation**  
   The clean numerical phase map is compared with the analytical transition references.

4. **Finite-size check**  
   Larger system sizes are used at selected points to assess finite-size effects.

5. **Noise response**  
   The evolution of phase-sensitive observables is tracked as the depolarizing probability is increased.

---

# Results

> **This section is intentionally kept separate for the final numerical results and figures.**

## Clean Phase Diagram

**[Insert clean ED phase diagram here]**

Add the 30 × 30 phase map, analytical reference curves, and a short interpretation of the observed phase regions.

## Noisy Phase Diagrams

### \(p = 0.01\)

**[Insert noisy phase diagram here]**

### \(p = 0.05\)

**[Insert noisy phase diagram here]**

Add a comparison showing how the observable-based phase structure changes with increasing noise.

## VQE Validation

**[Insert ED vs VQE energy comparison here]**

Include the representative parameter points and the corresponding energy agreement.

## Noise and Phase Robustness

**[Insert quantitative noise analysis here]**

Possible quantities to report:

- Boundary displacement
- Reduction in phase-sensitive correlations
- Relative robustness of different phases
- Change in antiphase / finite-\(q\) signatures
- Behavior of any floating/incommensurate candidate region

## Finite-Size Validation

**[Insert \(N=12\) validation results here]**

Use selected cuts, observables, or phase maps to show how the finite-size picture changes with system size.

---

## Key Contributions

The project combines:

- **Exact diagonalization** for a controlled classical reference
- **Observable-based phase classification**
- **VQE** as an independent quantum approach
- **Gate-level depolarizing noise modelling**
- **30 × 30 phase-space scanning**
- **Finite-size validation**
- **Accelerated circuit simulation with TensorCircuit-ng, JAX, and Optax**

This combination allows the clean phase structure, quantum-method agreement, and noise-induced degradation to be studied within one consistent workflow.

---

## Project Structure

```text
QSite/
├── starter_kit/
├── starter.ipynb
├── Exact Diagonalization Implementation.ipynb
├── README.md
└── results/
    ├── clean/
    ├── noisy/
    └── vqe/
