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

**Rubén Lucendo**  *AI Engineer & Product Builder*

Building systems that bridge the gap between theory and business value.

---

## Legal, Clinical, and Regulatory Disclaimer

**STRICTLY FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY. NOT FOR CLINICAL USE.**

The software, code, models, weights, algorithms, and any associated documentation provided in this repository (collectively referred to as the "Software") are experimental in nature and are intended strictly for academic, educational, and non-commercial research purposes. 

By accessing, downloading, or utilizing this Software, you explicitly acknowledge and agree to the following terms:

1. **No Medical Advice or Clinical Decision Support:** This Software does not constitute, nor is it intended to be a substitute for, professional medical advice, diagnosis, treatment, or clinical decision-making. The simulated macroscopic projections, spatiotemporal predictions, and any other outputs generated by the PINN (Physics-Informed Neural Network) or associated models are purely theoretical and have not been validated for clinical efficacy or accuracy. You must not rely on any output from this Software to make clinical or medical decisions regarding any patient.
2. **No Regulatory Clearance:** This Software has NOT been cleared, approved, or evaluated by the U.S. Food and Drug Administration (FDA), the European Medicines Agency (EMA), or any other global regulatory authority as a medical device or Software as a Medical Device (SaMD).
3. **Data Privacy and Compliance:** The user assumes full and sole responsibility for ensuring that any data (including but not limited to medical imaging, NIfTI files, or patient metadata) processed using this Software complies with all applicable local, state, national, and international data protection and privacy laws, including but not limited to the Health Insurance Portability and Accountability Act (HIPAA) in the United States and the General Data Protection Regulation (GDPR) in the European Union. The author(s) explicitly disclaim any responsibility for the unlawful or non-compliant use of Protected Health Information (PHI) or Personally Identifiable Information (PII) in conjunction with this Software.
4. **Limitation of Liability and Indemnification:** IN NO EVENT SHALL THE AUTHOR(S), CONTRIBUTORS, OR AFFILIATED ENTITIES BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS OR MEDICAL MALPRACTICE CLAIMS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION. By using this Software, you agree to indemnify, defend, and hold harmless the author(s) from any and all claims, liabilities, damages, and expenses (including legal fees) arising from your use or misuse of the Software.
5. **Disclaimer of Warranty:** THE SOFTWARE IS PROVIDED ON AN "AS IS" AND "AS AVAILABLE" BASIS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT. THE AUTHOR(S) MAKE NO REPRESENTATIONS THAT THE MACROSCOPIC DIGITAL TWIN SIMULATIONS OR PREDICTIONS WILL BE ACCURATE, ERROR-FREE, OR BIOLOGICALLY PLAUSIBLE.

Use of this repository constitutes your unconditional acceptance of these terms.