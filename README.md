# PINN-based Macroscopic Digital Twin for Glioblastoma Progression

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange)
![MONAI](https://img.shields.io/badge/MONAI-Medical_AI-darkgreen)
![PINN](https://img.shields.io/badge/Architecture-Physics_Informed_NN-8A2BE2)

## 1. Project summary

This repository, **GlioSim**, introduces a predictive spatiotemporal pipeline designed to act as a **Macroscopic Digital Twin** for Glioblastoma (GBM). Moving beyond static 3D segmentation, this system models the future physical growth of brain tumors by fusing Deep Learning with biophysical laws.

By leveraging a Physics-Informed Neural Network (PINN), the model takes an initial multi-modal MRI scan at $T_0$ and predicts the tumor's volumetric expansion and invasion into surrounding tissue at future timepoints ($T_n$), providing a critical macro-level foundation for in-silico clinical simulations.

---

## 2. Clinical Context & The "Digital Twin" Approach

Glioblastoma is highly aggressive, and understanding its patient-specific growth trajectory is vital for treatment planning. While traditional Deep Learning excels at recognizing patterns in static images, it struggles to predict biologically plausible future states without hallucinating.

To solve this, GlioSim does not just learn from data; it learns from physics. We constrain the neural network using the **Fisher-Kolmogorov reaction-diffusion equation**, the standard mathematical model for glioma growth:

$$\frac{\partial c}{\partial t} = \nabla \cdot (D \nabla c) + \rho c$$

Where:
* $c$ is the normalized tumor cell concentration.
* $D$ represents the diffusion tensor (tumor invasion into white/gray matter).
* $\rho$ is the proliferation rate of the active tumor core.

The network extracts the baseline geometry from the MRI and learns the patient-specific $D$ and $\rho$ parameters, allowing the engine to simulate forward in time.

---

## 3. Architecture & Methodology



The architecture is split into two decoupled sub-systems to ensure modularity:

1.  **State Extractor (3D U-Net):** Processes the raw `.nii.gz` multi-modal MRI (FLAIR, T1w, T1gd, T2w) at $T_0$ to establish the baseline spatial boundaries (Necrosis, Edema, Active Core).
2.  **Physics-Informed Simulator (PINN):** A temporal network that takes the spatial embeddings and a time parameter $t$ to generate the future tumor mask, penalized by a custom loss function that enforces the PDE (Partial Differential Equation) of tumor growth.

### Directory Structure
```text
gliosim/
├── configs/
│   ├── data_config.yaml         # Longitudinal data loading settings
│   └── pinn_config.yaml         # PDE hyperparameters and network sizing
├── data/                        # Ignored in git (UPenn-GBM / BraTS Longitudinal)
├── scripts/
│   ├── train_extractor.py       # Trains the spatial baseline network
│   └── train_simulator.py       # Trains the PINN with custom PDE loss
├── src/
│   ├── data/
│   │   └── longitudinal_dm.py   # LightningDataModule for T0 -> Tn pairs
│   ├── models/
│   │   ├── unet_baseline.py     # 3D Spatial Feature Extractor
│   │   └── pinn_simulator.py    # Spatiotemporal network
│   └── physics/
│       └── fisher_kolmogorov.py # PyTorch implementation of the PDE loss
└── requirements.txt
```

---

## 4. Dataset

This project utilizes longitudinal medical imaging datasets, specifically relying on multi-timepoint scans from the same patient to validate the temporal predictions.

* **Primary Source:** UPenn-GBM (University of Pennsylvania Glioblastoma Dataset) or BraTS Longitudinal.
* **Format:** 3D NIfTI volumes co-registered to the same anatomical space.

---

## 5. Evaluation Metrics

Evaluating a spatiotemporal simulation requires strict clinical and mathematical metrics:

* **Temporal Dice Score:** Measures the volumetric overlap between the simulated prediction at $T_n$ and the actual ground-truth MRI taken at $T_n$.
* **Volume Error Trajectory:** Calculates the difference in total predicted tumor volume (in $cm^3$) versus reality over a 3-to-6 month simulated curve.
* **PDE Residual Loss:** Quantifies how well the network's prediction obeys the underlying biophysical laws of diffusion and proliferation (lower is better).

---

## 6. How to use and replicate

### Prerequisites
* Git & Python 3.11+
* CUDA-enabled GPU (Minimum 16GB VRAM for 3D spatiotemporal backpropagation).

### Setup
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/rlucendo/gliosim.git
    cd gliosim
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run a Simulation (Inference):**
    Given a patient's baseline MRI, predict the tumor state 60 days into the future.
    ```bash
    python scripts/simulate.py \
        --input_mri data/patient_001_T0.nii.gz \
        --days_forward 60 \
        --output_dir simulations/
    ```

---

## Future Roadmap

* [ ] **White Matter Tractography Integration (DTI):** Enhancing the diffusion parameter ($D$) by incorporating anisotropic diffusion along white matter tracts.
* [ ] **Treatment Simulation (Radiotherapy):** Adding a decay term to the PDE to simulate tumor shrinkage after a simulated radiation dose.

---

## Author

**Rubén Lucendo** *AI Engineer & Product Builder*

Building systems that bridge the gap between theory and business value.