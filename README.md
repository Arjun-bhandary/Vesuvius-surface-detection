# 🌋 48th Place Solution: Vesuvius Challenge - Surface Detection

This repository contains the code, methodology, and inference pipeline for my **48th Place** finish in the [Kaggle Vesuvius Challenge - Surface Detection](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection) competition. 

## 📜 Competition Overview
The goal of this competition was to detect thin, crumpled papyrus surfaces within 3D micro-CT volumes of ancient Herculaneum scroll fragments. The objective is to produce highly accurate 3D binary segmentation masks that isolate the sheet structure, acting as the crucial first step for virtually unwrapping and reading the lost texts.

### 📐 Evaluation Metric
Unlike standard segmentation tasks, this competition heavily penalized topological errors (like artificial holes in a sheet or merged layers) rather than just voxel-wise misclassifications. The final leaderboard was ranked using a weighted combination of three topological and surface-aware metrics:

**`Score = 0.30 × TopoScore + 0.35 × SurfaceDice@2.0 + 0.35 × VOI_score`**

* **TopoScore:** Uses Betti matching to evaluate topological correctness. It severely penalizes false bridges/tunnels across adjacent wraps (Betti-1 errors) or artificial holes/breaks within a continuous sheet (Betti-0 errors).
* **SurfaceDice@2.0:** A spatial-tolerance variant of the Dice score. It checks if the predicted and ground truth surfaces lie within a physical spatial tolerance of 2.0 units of each other, allowing for slight geometrical shifts.
* **VOI_score (Variation of Information):** Evaluates instance-level consistency by measuring information loss, effectively penalizing over-segmentation (false splits) and under-segmentation (false merges).

## 🧠 Model Architecture & Strategy
I experimented with a wide variety of 3D segmentation models to tackle the complex geometry of the scrolls. Ultimately, the robust baselines established in medical imaging domains proved superior here. 

The best-performing approach utilized **nnU-Net**:
* **Architecture Setup:** `medM` `fullresenc` 
* **Patch Size:** `192` (Providing a strong balance between maintaining a wide spatial context and managing the memory constraints of 3D convolutions).

### 🏋️ Training Pipeline
To maximize both raw voxel accuracy and topological integrity, the network was trained using a two-phase schedule:
1.  **Phase 1 (Generalization):** `600` epochs utilizing the standard nnU-Net loss (typically a combination of Cross-Entropy and Dice loss) to learn the primary sheet structures.
2.  **Phase 2 (Topological Refinement):** `400` epochs fine-tuning with a **custom loss** function designed specifically to bridge gaps, prevent layer merges, and align the network's output with the rigorous demands of the TopoScore and VOI metrics.

## ⚙️ Inference & Post-Processing
Raw volumetric predictions from the model required careful refinement to handle the extreme sensitivity of the competition metric. Inference and post-processing are handled entirely within the provided Jupyter notebook.

**Post-Processing Pipeline:**
1.  **Hysteresis Thresholding:** Used instead of a hard global threshold to maintain the connectivity of faint, ambiguous papyrus structures. It leverages a high threshold to identify strong surface cores and a lower threshold to safely extend those components, preventing sheets from prematurely breaking apart.
2.  **Anisotropic Closing:** Applied structurally to bridge small, unnatural micro-gaps and holes in the predicted sheets, improving the Betti-1 score without accidentally fusing adjacent, tightly rolled layers.

## 🚀 How to Run
All inference steps, including the thresholding and morphological post-processing logic, are self-contained.

1. Ensure your environment has the necessary 3D segmentation dependencies installed (nnU-Net, PyTorch, SciPy, etc.).
2. Update the file paths in the provided `.ipynb` notebook to point to your local test volumes and your trained nnU-Net weights.
3. Run the inference notebook to generate the final post-processed 3D masks for submission.
