# LieOTMap: Rigid-Body Fitting of Atomic Structures into Cryo-EM Maps using Lie-Theoretic Optimal Transport

## Overview

LieOTMap is a high-accuracy software tool for rigidly fitting atomic models (from PDB or mmCIF files) into cryo-electron microscopy (cryo-EM) density maps. It formulates the fitting task as a continuous optimization problem that is differentiable from end-to-end, allowing for the use of powerful gradient-based optimization methods.

The core challenge in cryo-EM fitting is finding the correct 3D rotation and translation of a model to match the density map. Traditional methods often struggle with complex scoring landscapes or rely on non-differentiable metrics (like TM-score), which limits the optimization strategies that can be applied. LieOTMap overcomes these limitations by combining two powerful mathematical frameworks:

1.  **Lie Theory:** Rigid-body motion is parameterized on the Lie algebra $\mathfrak{se}(3)$. This provides a minimal, 6-dimensional, and singularity-free representation of the transformation, making it ideal for unconstrained gradient descent.
2.  **Optimal Transport (OT):** The dissimilarity between the atomic model and the density map (both treated as point clouds) is measured using the Sinkhorn divergence, a differentiable and computationally efficient proxy for the geometric Wasserstein distance. This serves as a robust loss function to guide the optimization.

By minimizing this loss function with respect to the Lie algebra parameters, LieOTMap can efficiently and accurately determine the optimal fit.

## Core Methodology

The algorithm proceeds in four main stages:

1.  **Point Cloud Generation:** The input atomic model is converted into a point cloud $X$ (typically using C-alpha atoms). The target cryo-EM map is converted into a point cloud $Y$ by selecting all voxels above a given sigma-level threshold (e.g., 3.0-sigma).

2.  **Lie Algebra Parameterization:** The rigid transformation $T$ is represented by a 6-dimensional vector $\xi \in \mathbb{R}^6$ in the Lie algebra $\mathfrak{se}(3)$. The corresponding matrix transformation $T = \exp(\hat{\xi})$ is recovered using the matrix exponential, ensuring a valid SE(3) transformation at every step.

3.  **Differentiable Scoring:** The algorithm computes the Sinkhorn divergence, $S_\epsilon(T(X), Y)$, between the transformed model point cloud and the target map point cloud. This score is fully differentiable with respect to the transformation parameters $\xi$.

4.  **Gradient-Based Optimization:** Using the Adam optimizer, the algorithm computes the gradient of the Sinkhorn divergence with respect to $\xi$ and iteratively updates the transformation to minimize the loss. This process effectively pulls the atomic model into the correct position and orientation within the density map.

## Usage and Parameters

The program is run from the command line. The primary script is `fitter_sinkhorn_final.py`.

### Main Parameters:

*   `--mobile_structure <path>`: **(Required)** Path to the PDB or mmCIF file of the atomic model you want to fit.

*   `--target_map <path>`: **(Required)** Path to the MRC file of the target cryo-EM density map.

*   `--gold_standard_structure <path>`: **(Optional)** Path to the known correct structure (in PDB or mmCIF format). If provided, the program will calculate and print the RMSD against this reference at each step, which is useful for validation and benchmarking.

*   `--sigma_cutoff <float>`: (Default: 3.0) The sigma level used to threshold the density map for generating the target point cloud. Higher values result in a sparser, higher-density point cloud.

*   `--learning_rate <float>`: (Default: 0.05) The learning rate for the Adam optimizer.

*   `--n_steps <int>`: (Default: 1000) The number of optimization steps to perform.

*   `--output_pdb <path>`: (Optional) The file path to write the final fitted PDB structure. If not provided, a default name will be generated based on the input and the final RMSD.

## Example Command Line

The following command was used to run the test case described in our paper, fitting the apo GroEL structure (1aon) into the ATP-bound map (EMD-1046) and validating against the known structure (1GRU).

```bash
python /Users/huyue/cryoem/fitter_sinkhorn_final.py \
    --mobile_structure /Users/huyue/cryoem/1aon.cif \
    --target_map /Users/huyue/cryoem/EMD-1046.map \
    --gold_standard_structure /Users/huyue/cryoem/1GRU.cif
```

This command will execute the full fitting pipeline and save the resulting structure as `1aon_sinkhorn_final_rmsd_3.08.pdb` (or similar, depending on the exact final RMSD).
