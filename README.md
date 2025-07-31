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
| Anomaly detection with robust deep autoencoders | AE                | Robust AE decomposes input into clean and noisy parts.  | ACM SIGKDD | 2017 |
| Spatio-temporal autoencoder for video anomaly detection | AE                | 3D convolutional spatio-temporal AE with weight-decreasing loss.                      | ACM MM | 2018 |
| Deep autoencoding gaussian mixture model for unsupervised anomaly detection | AE+GMM            | Jointly optimizes AE representations and reconstruction within a GMM.                 | ICLR | 2018 |
| Outlier detection for time series with recurrent autoencoder ensembles | Ensemble AEs      | Ensemble of sparsely-connected RNN-based AEs.                                         | IJCAI | 2019 |
| Memorizing normality to detect anomaly: Memory-augmented deep autoencoder for unsupervised anomaly detection | AE+Memory         | Learnable memory module with AE retrieves prototypical normal patterns.               | ICCV | 2019 |
| Learning memory-guided normality for anomaly detection | AE+Memory         | Memory-augmented AE with prototypical memory update.                                  | CVPR | 2020 |
| Auto-ad: Autonomous hyperspectral anomaly detection network based on fully convolutional autoencoder | AE                | Convolutional AE with skip connections and adaptive-weighted loss.                    | IEEE TGRS | 2022 |
| Self-supervised masked convolutional transformer block for anomaly detection | Masked AE         | Self-supervised masked convolutional transformer AE block.                            | IEEE TPAMI | 2023 |
| Anomaly detection for medical images using heterogeneous auto-encoder | Hybrid AE         | CNN-Transformer AE with multi-scale sparse attention.                                 | IEEE TIP | 2024 |
| Ada-gad: Anomaly-denoised autoencoders for graph anomaly detection | AE                | GNN-based AE with anomaly-denoised augmentation and multi-level pertaining.           | AAAI | 2024 |
| Rethinking reconstruction-based graph-level anomaly detection: limitations and a simple remedy | AE                | Uses multifaceted statistics of reconstruction errors as discriminative features.     | 	NeurIPS | 2024 |
| A recover-then-discriminate framework for robust anomaly detection | AE                | Recover-then-discriminate framework with prompted images for reconstruction.          | SCIS | 2025 |
| Winner-take-all autoencoders | SAE               | Adds sparsity penalty on activations.                                                 | NeurIPS | 2015 |
| Cloud intrusion detection method based on stacked contractive auto-encoder and support vector machine | Stacked CAE       | Stacked CAE for feature extraction with SVM-based classification.                     | IEEE TCC | 2020 |
| Memstream: Memory-based streaming anomaly detection | DAE+Memory        | Uses a FIFO memory module with DAE to adapt to evolving streaming data.               | WWW | 2022 |
| Meter: A dynamic concept adaptation framework for online anomaly detection | DAE+ hypernetwork | Combines evidential drift detection with a hypernetwork for parameter shift generation. | arXiv:2312.16831 | 2022 |
| Dictionary trained attention constrained low rank and sparse autoencoder for hyperspectral anomaly detection | SAE               | Attention-constrained SAE to capture spatial semantics.                               | Neural Networks | 2025 |
| Ensemble denoising autoencoders based on broad learning system for time-series anomaly detection | DAE               | DAE with broad learning system and progressive anomaly augmentation.                  | IEEE TNNLS | 2025 |

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

|   Paper Title   | Anomaly Score             | Key Idea                                 |     Venue     | Year | Ref.  |
|---------------------|---------------------------|----------------------------------------------|------------|------|----|
| Unsupervised anomaly detection with generative adversarial networks to guide marker discovery  | Hybrid score (recon&disc) | Early GAN-based method combining visual fidelity and discriminative feedback.  | IPMI  | 2017 |
| MAD-GAN [74]        | Hybrid score (recon&disc) | Ensemble discriminator scoring for robust time-series anomaly detection.                                      |   | 2019 |
| Liu et al. [119]    | Discriminator confidence  | Multi-generator strategy for outlier synthesis and a discriminator for detection.                             |   | 2019 |
| ATR-GAN [127]       | GAN-augmented classifier  | Time-regularized GAN with augmented filtering and temporal-aware distance constraints.                        |   | 2021 |
| RegraphGAN [118]    | Reconstruction error      | Encodes relational dependencies to detect structural graph deviations.                                        |   | 2023 |
| GANomaly [120]      | Reconstruction error      | Robust detection of unseen anomalies via joint latent and image space learning.                               |   | 2018 |
| OCGAN [121]         | Reconstruction error      | Constrains latent space to suppress outlier reconstruction for one-class anomaly detection.                   |   | 2019 |
| f-AnoGAN [122]      | Hybrid score (recon&disc) | Latent space mapping for fast anomaly localization in medical imaging.                                        |   | 2019 |
| GAN ensembles [123] | Hybrid score (recon&disc) | Uses an ensemble of GANs to enhance detection performance over various base models.                           |   | 2021 |
| Anomaly-GAN [126]   | GAN-augmented classifier  | Mask-guided GAN generates diverse anomalies with local-global discriminators for surface inspection.          |   | 2023 |
| ACSleuth [124]      | Reconstruction error      | Memory-augmented domain-adaptive GAN for fine-grained anomalous cell detection.                               |   | 2024 |
| Miao et al. [125]   | Reconstruction error      | Integrates contrastive learning and data augmentation into a transformer-based GAN for MTS anomaly detection. |   | 2024 |


#### 1.1.4 Diffusion-based Reconstruction Methods

|  Paper Title       | Technique         | Key Idea         |     Venue     | Year | Ref.  |
|---------------------|---------------------------|---------------------------------------|------------|-----|------|
| Anoddpm: Anomaly detection with denoising diffusion probabilistic models using simplex noise  | Conditional DDPM | Multi-scale simplex noise with partial diffusion to detect anomalies in high-resolution medical images.  | CVPR | 2022 |
| GLAD [134]           | DDIM             | Employs global and local reconstruction with deterministic noise scheduling for enhanced anomaly localization.                             |   | 2024 |
| R3D-AD [135]         | Conditional DDPM | Latent space regularization and intermediate-step perturbations for precise 3D anomaly detection.                                          |   | 2024 |
| DiffusionAD [133]    | DDPM             | Norm-guided single-step refinement with pixel-wise segmentation for improved anomaly localization.                                         |   | 2025 |
| DCMD [139]           | Conditional DDPM | Combines conditioned motion and embedding in a DDPM-based model with temporal-spatial associations for pose-based video anomaly detection. |   | 2025 |
| DiffAD [27]          | Conditional DDPM | Noisy conditioning embeddings and interpolation channels to diversify the reconstruction process and improve noise guidance mitigation.    |   | 2023 |
| DiAD [28]            | Conditional DDPM | Semantic feature fusion of multi-scale features, linking SD and the semantic-guided network, thereby maintaining semantic consistency.     |   | 2024 |
| Tebbe & Tayyub [138] | Conditional DDIM | Adaptively modulates the noising process based on anomaly estimates.                                                                       |   | 2024 |
| DZAD [139]           | Conditional DDPM | Utilizes noise features during the diffusion denoising process to enhance zero-shot generalization.                                        |   | 2025 |


#### 1.1.5 Hybrid Reconstruction Methods


### 1.2 Representation-driven Approaches

#### 1.2.1 LLM-based Methods

| Paper Title      | FM               | Data              | Scenario                                |     Venue     | Year | Ref.  |
|--------------------------|------------------|-------------------|-----------------------------------------|------|------|------|
| Logprompt: Prompt engineering towards zero-shot and interpretable log analysis   | GPT-3.5     | Log       | Software maintenance log     |ICSE-Companion   |  2024 |
| LogConfigLocalizer [158] | GPT-4            | Log               | Localize configuration errors           |   |  2024 |
| Audit-LLM [159]          | GPT-3.5          | Log               | Log-based insider threat detection      |   |  2024 |
| LogiCode [30]            | GPT-4            | Text              | Industrial logical text                 |   |  2024 |
| Spiced [160]             | GPT-3.5          | Signal            | Trojan detection in A/MS designs        |   |  2024 |
| Sinha et al. [161]       | GPT-4            | Text              | Robotic system monitor                  |   |  2024 |
| LLMAD [162]              | GPT-4            | Time series       | Univariate time series                  |   |  2024 |
| AnomalyLLM [155]         | GPT-2            | Time series       | Univariate and multivariate time series |   |  2024 |
| Dong et al. [163]        | GPT-4            | Time series       | Univariate time series                  |   |  2024 |
| FLEXLOG [164]            | GPT-3            | Log               | Software systems log                    |   |  2024 |
| SIGLLM [165]             | GPT-3.5          | Time series       | Univariate time series                  |   |  2024 |
| DABL [166]               | LLaMA-2          | Text              | Business semantic text                  |   |  2025 |


#### 1.2.2 MLLM-based Methods
| Paper Title      | FM               | Data              | Scenario                                |     Venue     | Year | Ref.  |
|--------------------------|------------------|-------------------|-----------------------------------------|------|------|------|
| Random word data augmentation with clip for zero-shot anomaly detection   | CLIP             | Text & Image      | Industrial image      | Arxiv  | 2023 |
| Myriad [168]             | GPT-3.5          | Text & Image      | Industrial image                        |   | 2023 |
| AnomalyCLIP [169]        | CLIP             | Text & Image      | Industrial and medical image            |   | 2023 |
| WinCLIP [35]             | OpenCLIP         | Text & Image      | Industrial image                        |   | 2023 |
| AnomalyGPT [32]          | Vicuna           | Text & Image      | Industrial image                        |   | 2024 |
| FiLo [170]               | CLIP             | Text & Image      | Industrial image                        |   | 2024 |
| PromptAD [171]           | CLIP             | Text & Image      | Industrial image                        |   | 2024 |
| Hawk [172]               | LLaMA-2          | Text & Image      | Surveillance image                      |   | 2024 |
| VAD-LLaMA [173]          | LLaMA            | Text & Video      | Surveillance and traffic scene video    |   | 2024 |
| AdaCLIP [174]            | CLIP             | Text & Image      | Industrial and medical image            |   | 2024 |
| Li et al. [175]          | CLIP             | Text & Image      | Industrial, medical, and semantic image |   | 2024 |
| InCTRL [176]             | OpenCLIP         | Text & Image      | Industrial, medical, and semantic image |   | 2024 |
| CLIP-AD [157]            | CLIP             | Text & Image      | Industrial and medical image            |   | 2024 |
| MVFA [12]                | CLIP             | Text & Image      | Medical image                           |   | 2024 |
| VERA [177]               | InternVL2        | Text & Image      | Surveillance image                      |   | 2025 |
| Holmes-VAU [178]         | InternVL2        | Text & Video      | Surveillance video                      |   | 2025 |
| UniVAD [179]             | CLIP             | Text & Image      | Industrial, logical, and medical image  |   | 2025 |
| AA-CLIP [180]            | CLIP             | Text & Image      | Industrial and medical image            |   | 2025 |
| Bayes-PFL [181]          | CLIP             | Text & Image      | Industrial and medical image            |   | 2025 |
| EchoTraffic [182]        | LLaMA-2          | Text, Video&Audio | Traffic anomaly understanding           |   | 2025 |

#### 1.2.3 Hybrid Representation-driven Methods

| Paper Title      | FM               | Data              | Scenario                                |     Venue     | Year | Ref.  |
|--------------------------|------------------|-------------------|-----------------------------------------|------|------|------|
| Unsupervised video anomaly detection based on similarity with predefined text descriptions   | ChatGPT & CLIP   | Text & Image      | Surveillance image   |  Sensors | 2023 |
| LAVAD [31]               | LLaMA-2 & BLIP-2 | Text & Video      | Surveillance video                      |   | 2024 |
| AnomalyRuler [184]       | GPT-4 & Mistral  | Text & Video      | Surveillance video                      |   | 2024 |
| ALFA [29]                | GPT-3.5 & CLIP   | Text & Image      | Industrial image                        |   | 2024 |


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
