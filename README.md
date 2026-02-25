# PINN-based Macroscopic Digital Twin for Glioblastoma Progression

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange)
![MONAI](https://img.shields.io/badge/MONAI-Medical_AI-darkgreen)
![PINN](https://img.shields.io/badge/Architecture-Physics_Informed_NN-8A2BE2)

## 1. Project summary

This repository, **GlioSim**, introduces a predictive spatiotemporal pipeline designed to act as the macroscopic foundation for a **Glioblastoma (GBM) Digital Twin**. Moving beyond the paradigm of static 3D anatomical segmentation, this system models the future physical growth, diffusion, and invasion of brain tumors by fusing Deep Learning with established biophysical laws.

By leveraging a **Physics-Informed Neural Network (PINN)**, GlioSim ingests an initial multi-modal MRI scan (FLAIR, T1w, T1gd, T2w) at a baseline timepoint ($T_0$) and predicts the tumor's volumetric expansion at arbitrary future timepoints ($T_n$). This engine provides the critical macro-level topological state required to initialize and constrain high-fidelity, multi-omics in-silico clinical simulations, effectively bridging the gap between macroscopic radiology and cellular-level disease modeling.

---

## 2. Clinical Context & The "Digital Twin" Approach

### The Challenge of Glioblastoma Progression
Glioblastoma is the most aggressive primary brain tumor, characterized by rapid cellular proliferation and diffuse infiltration into the surrounding healthy brain parenchyma (white and gray matter). Traditional Deep Learning models in medical imaging are primarily *descriptive*—they excel at segmenting visible tumor boundaries at a single point in time. However, to build a true **Digital Twin** for in-silico treatment planning, the system must be *predictive*. 

Standard, purely data-driven autoregressive models (like 3D ConvLSTMs) often fail in this domain. They tend to "hallucinate" biologically impossible tumor shapes when extrapolating forward in time because they lack an understanding of the underlying physical constraints of the human brain.

### The Physics-Informed Solution
To prevent these biological hallucinations, GlioSim constrains the neural network's predictions using the **Fisher-Kolmogorov reaction-diffusion equation**, the gold-standard macroscopic mathematical model for glioma progression. 

Instead of letting the network guess the next frame, we force it to minimize a loss function that satisfies the following Partial Differential Equation (PDE):

$$\frac{\partial u}{\partial t} = \nabla \cdot (D \nabla u) + \rho u (1 - u)$$

Where:
* $u(\mathbf{x}, t)$ is the normalized tumor cell concentration at 3D spatial coordinate $\mathbf{x}$ and time $t$.
* $D(\mathbf{x})$ represents the spatial diffusion tensor (modeling the tumor's invasive spread, which moves faster through white matter than gray matter).
* $\rho(\mathbf{x})$ is the localized cellular proliferation rate.
* The term $(1 - u)$ enforces logistic growth, representing the carrying capacity of the tissue (preventing infinite biological density).

In this architecture, the PINN acts as an inverse-problem solver. It does not just output a future image; it implicitly learns the patient-specific spatial distributions of $D$ and $\rho$ from the baseline MRI, creating a truly personalized predictive model.

---

## 3. Architecture & Methodology



To manage the immense memory footprint of 3D NIfTI tensors and the complexity of calculating spatiotemporal gradients (required for the PDE loss), the architecture is decoupled into two primary subsystems:

### Phase A: The Baseline State Extractor ($T_0$)
A customized 3D U-Net (implemented via MONAI) processes the raw, multi-modal NIfTI volumes. Its objective is to map the radiological intensities into a normalized biological state representation $u(\mathbf{x}, 0)$. It extracts the initial spatial boundaries of the necrotic core, the enhancing tumor, and the peritumoral edema, projecting them into a latent anatomical embedding.

### Phase B: The PINN Simulator ($T_0 \rightarrow T_n$)
The core simulation engine. It takes the spatial embedding from Phase A and a continuous time parameter $t$ as inputs. 

The network is trained using a composite loss function:
$$L_{Total} = L_{Data} + \lambda L_{PDE}$$

* **$L_{Data}$ (Data Fidelity):** Ensures the prediction matches available longitudinal ground-truth MRIs (e.g., forcing the prediction at $t=30$ days to match the actual scan taken at day 30).
* **$L_{PDE}$ (Physics Penalty):** Uses PyTorch's `autograd` to compute the spatial ($\nabla u$) and temporal ($\frac{\partial u}{\partial t}$) derivatives of the network's output. It penalizes any prediction that violates the Fisher-Kolmogorov equation, ensuring the generated tumor grows biologically, even when predicting into the unseen future.

### System Directory Structure
The codebase follows Domain-Driven Design principles, isolating the mathematical physics from the data engineering and model serving infrastructure.

```text
gliosim/
├── configs/
│   ├── data_config.yaml         # ETL settings, caching, and MONAI transforms
│   └── pinn_config.yaml         # PDE hyperparameters (lambda weights) and sizing
├── data/                        # Local data directory (Ignored in git)
├── scripts/
│   ├── train_extractor.py       # Pipeline for the 3D U-Net baseline
│   ├── train_simulator.py       # PINN training loop with continuous time-sampling
│   └── simulate.py              # Inference script: generates future NIfTI volumes
├── src/
│   ├── data/
│   │   └── longitudinal_dm.py   # LightningDataModule handling T0->Tn pairs
│   ├── models/
│   │   ├── unet_baseline.py     # Feature extraction architecture
│   │   └── pinn_simulator.py    # Spatiotemporal network with time-conditioning
│   └── physics/
│       └── fisher_kolmogorov.py # Core PDE loss utilizing torch.autograd
└── pyproject.toml               # Modern, reproducible dependency management
```

## 4. Dataset & Preprocessing Requirements



Training a spatiotemporal model requires highly structured, longitudinal data. Unlike static segmentation, temporal models are extremely sensitive to spatial misalignment; if the brain shifts between $T_0$ and $T_n$, the PINN will erroneously learn that shift as "tumor diffusion."

### Recommended Data Sources
* **UPenn-GBM (TCIA):** The University of Pennsylvania Glioblastoma dataset provides multi-parametric MRI (mpMRI), genomic data, and clinical outcomes for over 600 patients, with longitudinal follow-ups.
* **BraTS-GLI Longitudinal:** Recent iterations of the BraTS challenge provide paired preoperative and postoperative/follow-up scans.

### Strict ETL Pipeline (The MONAI Advantage)
To ensure the PDE solver calculates valid spatial gradients ($\nabla u$), the input tensors must be geometrically flawless. The data pipeline enforces:
1.  **Rigid Co-registration:** All longitudinal scans ($T_n$) must be strictly co-registered to the patient's baseline ($T_0$) anatomical space (e.g., SRI24 atlas) using affine transformations.
2.  **Isotropic Resampling:** Voxel spacing must be normalized to exactly $1.0 \times 1.0 \times 1.0 \text{ mm}^3$ to ensure the diffusion tensor $D(\mathbf{x})$ computes physical distance accurately across all axes.
3.  **Skull-Stripping & Intensity Normalization:** Removing non-brain tissue and applying Z-score normalization per modality to ensure stable gradient flow during backpropagation.

---

## 5. Evaluation Metrics

Evaluating a spatiotemporal simulation requires moving beyond standard overlapping metrics (like a single Dice score) and assessing the dynamic trajectory of the prediction.

### Clinical & Topological Metrics
* **Temporal Dice Score (t-Dice):** Measures the volumetric overlap between the predicted tumor mask at simulated time $t$ and the actual ground-truth MRI taken at that specific time.
* **Volume Trajectory Error (VTE):** Calculates the Mean Absolute Error (MAE) between the predicted total tumor volume (in $cm^3$) and the actual volume over a longitudinal curve (e.g., day 0, day 30, day 90).
* **Center of Mass (CoM) Displacement:** Measures the Euclidean distance (in millimeters) between the centroid of the predicted tumor and the real tumor at $T_n$, validating if the simulated *direction* of invasion is correct.

### Mathematical Physics Metrics
* **PDE Residual Loss ($L_{PDE}$):** Quantifies how strictly the network's predictions obey the Fisher-Kolmogorov equation across the spatiotemporal domain. A lower residual indicates a more biologically plausible simulation, preventing impossible "teleportation" of tumor mass.

---

## 6. How to use and replicate

### Prerequisites
* Git & Python 3.11+
* NVIDIA GPU with CUDA 12.1+ (Minimum 16GB VRAM, e.g., RTX 4080, A10g, or A100 is highly recommended due to the heavy memory footprint of 3D gradient computation).

### Setup and Execution
1.  **Clone the repository and install dependencies:**
    ```bash
    git clone https://github.com/rlucendo/gliosim.git
    cd gliosim
    pip install -r requirements.txt
    ```

2.  **Phase A: Train the Baseline State Extractor:**
    ```bash
    python scripts/train_extractor.py --config configs/data_config.yaml
    ```

3.  **Phase B: Train the PINN Simulator:**
    ```bash
    python scripts/train_simulator.py --config configs/pinn_config.yaml
    ```

4.  **Inference (Simulate the Future):**
    Run the Digital Twin engine on a new patient to predict the tumor state 90 days forward.
    ```bash
    python scripts/simulate.py \
        --input_mri data/patient_X_T0.nii.gz \
        --days_forward 90 \
        --output_dir simulations/
    ```

---

## 7. Future roadmap

To evolve this macroscopic engine into a comprehensive Digital Twin ready for spatial multi-omics fusion, the following architectural upgrades are planned:

* [ ] **Anisotropic Diffusion via DTI:** Tumors do not grow equally in all directions; they travel faster along white matter tracts. Integrating Diffusion Tensor Imaging (DTI) into the PDE will transform the scalar diffusion parameter $D$ into an anisotropic tensor $\mathbf{D}$, significantly improving directional invasion accuracy.
* [ ] **Therapeutic Perturbation Modeling:** Upgrading the PDE to include a treatment decay term ($-\gamma u$) to simulate tumor shrinkage in response to localized radiotherapy or surgical resection masks.
* [ ] **Treatment Simulation (Radiotherapy):** Adding a decay term to the PDE to simulate tumor shrinkage after a simulated radiation dose.

---

## Author

**Rubén Lucendo**  
*AI Engineer & Product Builder*

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