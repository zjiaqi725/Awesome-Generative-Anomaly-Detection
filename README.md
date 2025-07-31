# Awesome-Generative-Anomaly-Detection
<p align="center">
  <img src="https://github.com/zjiaqi725/Awesome-Generative-Anomaly-Detection/blob/main/assets/fig_stat.png" width="1000">  
  <br>
  <em>Figure 1. Emerging Trends in Anomaly Detection Research: Keyword Frequencies and the Rising Influence of Generative AI. 
    Data are derived from the Web of Science database, with statistics collected on July 30, 2025. The word cloud visualizes the top 100 most frequent keywords extracted from 4152 anomaly detection papers published in 2025.
</p>


## 🔍 What is Generative Anomaly Detection?

Anomaly detection aims to identify data instances that deviate from expected patterns. Generative AI, including autoencoders, GANs, diffusion models, and foundation models, offers powerful tools for modeling complex data distributions. By learning to reconstruct, represent, or estimate the likelihood of normal data, these models enable the detection of subtle and high‑dimensional anomalies across diverse modalities such as images, videos, time series, and logs.

Generative approaches not only improve detection accuracy but also facilitate interpretability (e.g., anomaly localization) and complementary tasks such as anomaly synthesis and restoration, making them a promising direction for next‑generation anomaly detection systems.

## 📂 This repository collects:

A curated collection of generative AI–based anomaly detection methods, complementary generative tasks, and their key applications.

<p align="center">
  <img src="https://github.com/zjiaqi725/Awesome-Generative-Anomaly-Detection/blob/main/assets/fig_overview.png" width="1000">  
  <br>
  <em>Figure 2. A unified perspective on Generative Anomaly Detection.
</p>
    
### Table of Contents

- [1 Taxonomy of Generative Anomaly Detection Methods](#1-Generative-Anomaly-Detection-Methods)
  - [1.1 Reconstruction‑driven Methods](#11-reconstruction-driven-methods)
    - [1.1.1 AE Family-based Reconstruction Methods](#111-ae-family-based-reconstruction-methods)
    - [1.1.2 VAE-based Reconstruction Methods](#112-vae-based-reconstruction-methods)
    - [1.1.3 GAN-based Reconstruction Methods](#113-gan-based-reconstruction-methods)
    - [1.1.4 Diffusion-based Reconstruction Methods](#114-diffusion-based-reconstruction-methods)
    - [1.1.5 Hybrid Reconstruction Methods](#115-hybrid-reconstruction-methods)
  - [1.2 Representation‑driven Approaches](#12-representation-driven-approaches)
    - [1.2.1 LLM-based Methods](#121-llm-based-methods)
    - [1.2.2 MLLM-based Methods](#122-mllm-based-methods)
    - [1.2.3 Hybrid Representation-driven Methods](#123-hybrid-representation-driven-methods)
  - [1.3 Density Estimation‑driven Approaches](#density-estimation-driven-approaches)
    - [1.3.1 Explicit Density-based Methods](#131-explicit-density-based-methods)
    - [1.3.2 Implicit Density-based Methods](#132-implicit-density-based-methods)
- [2 Complementary Generative Tasks](#2-complementary-generative-tasks)
  - [2.1 Generative anomaly synthesis](#Generative-anomaly-synthesis)
  - [2.2 Generative anomaly restoration](#Generative-anomaly-restoration)
- [3 Applications](#3-applications)
  - [3.1 Industrial manufacturing](#Industrial-manufacturing)
  - [3.2 Medical field](#Medical-field)
  - [3.3 Finance market](#Finance-market)
  - [3.4 Cybersecurity](#Cybersecurity)

-----------------

## 1 Generative Anomaly Detection Methods

### 1.1 Reconstruction-driven Methods

#### 1.1.1 AE Family-based Reconstruction Methods

| Paper Title                       | Technique         | Key Idea                                                                              | Venue      | Year   | Ref. |
|-----------------------------------|-------------------|---------------------------------------------------------------------------------------|-----------|---------|------|
| Anomaly detection using autoencoders with nonlinear dimensionality reduction | AE                | Basic AE reconstructs input and detects anomalies via reconstruction error.  | MLSDA | 2014 |
| Anomaly detection with robust deep autoencoders | AE                | Robust AE decomposes input into clean and noisy parts.  | ACM SIGKDD   | 2017 |
| STAE [93]             | AE                | 3D convolutional spatio-temporal AE with weight-decreasing loss.                      | Captures spatio-temporal dynamics for video anomaly detection.                                    | 2018 |
| DAGMM [89]            | AE+GMM            | Jointly optimizes AE representations and reconstruction within a GMM.                 | Unifies reconstruction, probabilistic modeling, and regularization.                               | 2018 |
| Kieu et al. [101]     | Ensemble AEs      | Ensemble of sparsely-connected RNN-based AEs.                                         | Reduces overfitting to outliers by ensemble diversification.                                      | 2019 |
| MemAE [72]            | AE+Memory         | Learnable memory module with AE retrieves prototypical normal patterns.               | Restricts generalization to anomalies.                                                            | 2019 |
| Park et al. [90]      | AE+Memory         | Memory-augmented AE with prototypical memory update.                                  | Improves anomaly localization and mitigates over-generalization.                                  | 2020 |
| Auto-AD [94]          | AE                | Convolutional AE with skip connections and adaptive-weighted loss.                    | Enhances motion-sensitive anomaly reconstruction in videos.                                       | 2022 |
| SSMCTB [97]           | Masked AE         | Self-supervised masked convolutional transformer AE block.                            | Enables versatile plug-and-play self-supervised reconstruction.                                   | 2023 |
| Hetero-AE [99]        | Hybrid AE         | CNN-Transformer AE with multi-scale sparse attention.                                 | Addresses overfitting and saliency in medical anomaly detection.                                  | 2024 |
| ADA-GAD [19]          | AE                | GNN-based AE with anomaly-denoised augmentation and multi-level pertaining.           | Mitigates overfitting and the Homophily Trap in the graph.                                        | 2024 |
| MUSE [100]            | AE                | Uses multifaceted statistics of reconstruction errors as discriminative features.     | Reveals and remedies reconstruction flip via richer error statistics.                             | 2024 |
| ReDi [95]             | AE                | Recover-then-discriminate framework with prompted images for reconstruction.          | Resolves reconstruction ambiguity and feature-level inconsistency.                                | 2025 |
| Makhzani & Frey [86]  | SAE               | Adds sparsity penalty on activations.                                                 | Enhances focus on salient features.                                                               | 2015 |
| Wang et al. [92]      | Stacked CAE       | Stacked CAE for feature extraction with SVM-based classification.                     | Enhances intrusion detection by learning robust, compact features.                                | 2020 |
| MemStream [91]        | DAE+Memory        | Uses a FIFO memory module with DAE to adapt to evolving streaming data.               | Enables drift-resilient detection via online memory update.                                       | 2022 |
| METER [4]             | DAE+ hypernetwork | Combines evidential drift detection with a hypernetwork for parameter shift generation. | Supports interpretable and efficient adaptation to concept drift without retraining.              | 2022 |
| Hu et al. [96]        | SAE               | Attention-constrained SAE to capture spatial semantics.                               | Achieves joint background and anomaly reconstruction via dual-path encoding and nonlinear fusion. | 2025 |
| DBLS-AE [102]         | DAE               | DAE with broad learning system and progressive anomaly augmentation.                  | Enhances robustness for complex temporal anomalies.                                               | 2025 |

#### 1.1.2 VAE-based Reconstruction Methods

| Paper Title                  | Key Idea                                                                                                   | Venue     | Year | Ref.|
|-----------------------|------------------------------------------------------------------------------------------------------------|------------------|------|-----|
| Variational autoencoder-based anomaly detection using reconstruction probability | Utilizes reconstruction probability from VAE as anomaly score.    | Special lecture on IE    | 2015 |
| S²-VAE [105]          | Two-stage VAE with GMM for fast filtering and hierarchical feature learning.                               | Captures both local and global anomalies in video surveillance.                                     | 2018 |
| OmniAnomaly [106]     | Stochastic VAE capturing robust MTS representations with variable connection and planar normalizing flow.  | Enables interpretable, noise-resilient detection in various industry devices.                       | 2019 |
| GMFC-VAE [107]        | Two-stream GMM-VAE learning normal representations for patch-wise scoring.                                 | Fuses spatial-temporal cues via dynamic flows for video anomaly detection.                          | 2020 |
| DVGCRN [108]          | Deep variational graph convolutional recurrent network to model hierarchical spatiotemporal MTS structure. | Unifies forecasting and reconstruction in a layered variational architecture.                       | 2022 |
| MACE [109]            | Frequency-domain method featuring multi-normal pattern modeling.                                           | Models service-specific normal pattern for short-term anomaly detection.                            | 2024 |
| Tri-VAE [73]          | Triplet VAE enforcing lesion-normal separation via disentangled metric learning.                           | Improves lesion detection by semantic-guided reconstruction.                                        | 2024 |
| Zimmerer et al. [110] | Pixel-wise KL-divergence for robust anomaly scoring.                                                       | Generalizes across tasks without model-specific tuning.                                             | 2018 |
| SISVAE [111]          | Sequential VAE with smoothness-inducing prior for latent temporal consistency.                             | Enhances robustness to noise and non-stationarity in MTS anomaly detection.                         | 2020 |
| Chauhan et al. [112]  | Post hoc likelihood debiasing via bias correction and contrast normalization.                              | Mitigates image feature bias in VAE scoring to improve outlier detection.                           | 2022 |
| VQRAEs [77]           | Quasi-recurrent VAE with robust α-, β-, and γ-divergence.                                                  | Prevents anomaly contamination via divergence-regularized latent space.                             | 2022 |
| Deep SVDD-VAE [113]   | Joint VAE-SVDD learning in latent hypersphere.                                                             | Improves detection by aligning generation with discriminative boundaries.                           | 2020 |
| SLA-VAE [114]         | Semi-supervised VAE updated via uncertainty-guided active learning.                                        | Reduces labeling cost while maintaining adaptability in dynamic cloud settings.                     | 2022 |
| ACVAE [115]           | Contrastive VAE with periodic joint training and query model.                                              | Enhances MTS anomaly detection by considering out-of-band information in large distributed systems. | 2022 |
| TraceVAE [22]         | Dual-variable VAE with dispatch learning of structure and time features.                                   | Enhances trace-level detection by mitigating entropy-induced scoring bias.                          | 2024 |
| VAEAT [116]           | Adversarial VAE with attention and dual-decoder regularization.                                            | Captures complex temporal dependencies and resists noise in MTS.                                    | 2024 |


#### 1.1.3 GAN-based Reconstruction Methods


#### 1.1.4 Diffusion-based Reconstruction Methods


#### 1.1.5 Hybrid Reconstruction Methods


### 1.2 Representation-driven Approaches

#### 1.2.1 LLM-based Methods

#### 1.2.2 MLLM-based Methods

#### 1.2.3 Hybrid Representation-driven Methods


### 1.3 Density Estimation-driven Approaches

#### 1.3.1 Explicit Density-based Methods


#### 1.3.2 Implicit Density-based Methods




## 2 Complementary Generative Tasks

### 2.1 Generative anomaly synthesis
...

### 2.2 Generative anomaly restoration
...



## 3 Applications

### 3.1 Industrial manufacturing
...

### 3.2 Medical field
...

### 3.3 Finance market

...

### 3.4 Cybersecurity

...
