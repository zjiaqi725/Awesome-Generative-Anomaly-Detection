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

Reconstruction-driven approaches have emerged as a mainstream paradigm in generative anomaly detection, with demonstrated success across diverse data modalities.
The core idea of reconstruction-based anomaly detection is to train a generative model to capture the underlying normal data distribution and reconstruct in-distribution instances well.
Anomalies are then detected by measuring the reconstruction error, i.e., the difference between the input and its reconstruction, under the assumption that abnormal samples cannot be precisely reconstructed due to their deviation from learned normal patterns.

#### 1.1.1 AE Family-based Reconstruction Methods

| Paper Title                       | Technique         | Key Idea                                                                              | Venue      | Year   | Ref. |
|-----------------------------------|-------------------|---------------------------------------------------------------------------------------|-----------|---------|------|
| Anomaly detection using autoencoders with nonlinear dimensionality reduction | AE                | Basic AE reconstructs input and detects anomalies via reconstruction error.  | MLSDA | 2014 | [[paper]](https://dl.acm.org/doi/abs/10.1145/2689746.2689747) |
| Anomaly detection with robust deep autoencoders | AE                | Robust AE decomposes input into clean and noisy parts.  | ACM SIGKDD | 2017 | [[paper]](https://www.eecs.yorku.ca/course_archive/2018-19/F/6412/reading/kdd17p665.pdf) |
| Spatio-temporal autoencoder for video anomaly detection | AE                | 3D convolutional spatio-temporal AE with weight-decreasing loss.                      | ACM MM | 2018 |[[paper]](https://dl.acm.org/doi/abs/10.1145/3123266.3123451) |
| Deep autoencoding gaussian mixture model for unsupervised anomaly detection | AE+GMM            | Jointly optimizes AE representations and reconstruction within a GMM.                 | ICLR | 2018 |[[paper]](https://bzong.github.io/doc/iclr18-dagmm.pdf) |
| Outlier detection for time series with recurrent autoencoder ensembles | Ensemble AEs      | Ensemble of sparsely-connected RNN-based AEs.                                         | IJCAI | 2019 |[[paper]](https://homes.cs.aau.dk/~byang/papers/IJCAI2019.pdf) |
| Memorizing normality to detect anomaly: Memory-augmented deep autoencoder for unsupervised anomaly detection | AE+Memory         | Learnable memory module with AE retrieves prototypical normal patterns.               | ICCV | 2019 |[[paper]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gong_Memorizing_Normality_to_Detect_Anomaly_Memory-Augmented_Deep_Autoencoder_for_Unsupervised_ICCV_2019_paper.pdf) |
| Learning memory-guided normality for anomaly detection | AE+Memory         | Memory-augmented AE with prototypical memory update.                                  | CVPR | 2020 |[[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Park_Learning_Memory-Guided_Normality_for_Anomaly_Detection_CVPR_2020_paper.pdf) |
| Auto-ad: Autonomous hyperspectral anomaly detection network based on fully convolutional autoencoder | AE                | Convolutional AE with skip connections and adaptive-weighted loss.                    | IEEE TGRS | 2022 |[[paper]](https://ieeexplore.ieee.org/abstract/document/9382262) |
| Self-supervised masked convolutional transformer block for anomaly detection | Masked AE         | Self-supervised masked convolutional transformer AE block.                            | IEEE TPAMI | 2023 |[[paper]](https://ieeexplore.ieee.org/abstract/document/10273635) |
| Anomaly detection for medical images using heterogeneous auto-encoder | Hybrid AE         | CNN-Transformer AE with multi-scale sparse attention.                                 | IEEE TIP | 2024 |[[paper]](https://ieeexplore.ieee.org/abstract/document/10486200) |
| Ada-gad: Anomaly-denoised autoencoders for graph anomaly detection | AE                | GNN-based AE with anomaly-denoised augmentation and multi-level pertaining.           | AAAI | 2024 |[[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/28691) |
| Rethinking reconstruction-based graph-level anomaly detection: limitations and a simple remedy | AE                | Uses multifaceted statistics of reconstruction errors as discriminative features.     | 	NeurIPS | 2024 |[[paper]](https://proceedings.neurips.cc/paper_files/paper/2024/file/ae03bdef276132fae089692445725635-Paper-Conference.pdf) |
| A recover-then-discriminate framework for robust anomaly detection | AE                | Recover-then-discriminate framework with prompted images for reconstruction.          | SCIS | 2025 |[[paper]](https://link.springer.com/article/10.1007/s11432-024-4291-4) |
| Winner-take-all autoencoders | SAE               | Adds sparsity penalty on activations.                                                 | NeurIPS | 2015 |[[paper]]() |
| Cloud intrusion detection method based on stacked contractive auto-encoder and support vector machine | Stacked CAE       | Stacked CAE for feature extraction with SVM-based classification.                     | IEEE TCC | 2020 |[[paper]](https://ieeexplore.ieee.org/abstract/document/9112664) |
| Memstream: Memory-based streaming anomaly detection | DAE+Memory        | Uses a FIFO memory module with DAE to adapt to evolving streaming data.               | WWW | 2022 |[[paper]](https://dl.acm.org/doi/abs/10.1145/3485447.3512221) |
| Meter: A dynamic concept adaptation framework for online anomaly detection | DAE+ hypernetwork | Combines evidential drift detection with a hypernetwork for parameter shift generation. | Arxiv | 2022 |[[paper]](https://dl.acm.org/doi/abs/10.14778/3636218.3636233) |
| Dictionary trained attention constrained low rank and sparse autoencoder for hyperspectral anomaly detection | SAE               | Attention-constrained SAE to capture spatial semantics.                               | Neural Networks | 2025 |[[paper]](https://www.sciencedirect.com/science/article/pii/S0893608024007214) |
| Ensemble denoising autoencoders based on broad learning system for time-series anomaly detection | DAE               | DAE with broad learning system and progressive anomaly augmentation.                  | IEEE TNNLS | 2025 |[[paper]](https://ieeexplore.ieee.org/abstract/document/10937928) |

#### 1.1.2 VAE-based Reconstruction Methods

| Paper Title                  | Key Idea                                                                                                   | Venue     | Year | Ref.|
|-----------------------|------------------------------------------------------------------------------------------------------------|------------------|------|-----|
| Variational autoencoder-based anomaly detection using reconstruction probability | Utilizes reconstruction probability from VAE as anomaly score.    | Special lecture on IE    | 2015 |
| Generative neural networks for anomaly detection in crowded scenes | Two-stage VAE with GMM for fast filtering and hierarchical feature learning.                               | IEEE TIFS | 2018 |
| Robust anomaly detection for multivariate time series through stochastic recurrent neural network | Stochastic VAE capturing robust MTS representations with variable connection and planar normalizing flow.  | 	SIGKDD | 2019 |
| Video anomaly detection and localization via Gaussian mixture fully convolutional variational autoencoder | Two-stream GMM-VAE learning normal representations for patch-wise scoring.                                 | CVIU | 2020 |
| Deep variational graph convolutional recurrent network for multivariate time series anomaly detection | Deep variational graph convolutional recurrent network to model hierarchical spatiotemporal MTS structure. | ICML | 2022 |
| Learning multi-pattern normalities in the frequency domain for efficient time series anomaly detection | Frequency-domain method featuring multi-normal pattern modeling.                                           | ICDE | 2024 |
| Tri-vae: Triplet variational autoencoder for unsupervised anomaly detection in brain tumor mri | Triplet VAE enforcing lesion-normal separation via disentangled metric learning.                           | CVPR | 2024 |
|  Unsupervised anomaly localization using variational auto-encoders | Pixel-wise KL-divergence for robust anomaly scoring.                                                       | MICCAI | 2018 |
| Anomaly detection of time series with smoothness-inducing sequential variational auto-encoder | Sequential VAE with smoothness-inducing prior for latent temporal consistency.                             | IEEE TNNLS | 2020 |
|  Robust outlier detection by de-biasing vae likelihoods | Post hoc likelihood debiasing via bias correction and contrast normalization.                              | CVPR | 2022 |
|  Anomaly detection in time series with robust variational quasi-recurrent autoencoders | Quasi-recurrent VAE with robust α-, β-, and γ-divergence.                                                  | ICDE | 2022 |
| Vae-based deep svdd for anomaly detection | Joint VAE-SVDD learning in latent hypersphere.                                                             | Neurocomputing | 2020 |
|  A semi-supervised vae based active anomaly detection framework in multivariate time series for online systems | Semi-supervised VAE updated via uncertainty-guided active learning.                                        | WWW | 2022 |
| Situation-aware multivariate time series anomaly detection through active learning and contrast vae-based models in large distributed systems | Contrastive VAE with periodic joint training and query model.                                              | IEEE JSAC | 2022 |
| Unsupervised anomaly detection on microservice traces through graph vae | Dual-variable VAE with dispatch learning of structure and time features.                                   | WWW | 2024 |
| Vaeat: Variational autoeencoder with adversarial training for multivariate time series anomaly detection | Adversarial VAE with attention and dual-decoder regularization.                                            |  Information Sciences | 2024 |


#### 1.1.3 GAN-based Reconstruction Methods

|   Paper Title   | Anomaly Score             | Key Idea                                 |     Venue     | Year | Ref.  |
|---------------------|---------------------------|----------------------------------------------|------------|------|----|
| Unsupervised anomaly detection with generative adversarial networks to guide marker discovery  | Hybrid score (recon&disc) | Early GAN-based method combining visual fidelity and discriminative feedback.  | IPMI  | 2017 |
|  Mad-gan: Multivariate anomaly detection for time series data with generative adversarial networks | Hybrid score (recon&disc) | Ensemble discriminator scoring for robust time-series anomaly detection.                                      | ICANN | 2019 |
| Generative adversarial active learning for unsupervised outlier detection | Discriminator confidence  | Multi-generator strategy for outlier synthesis and a discriminator for detection.                             | IEEE TKDE  | 2019 |
| Augmented time regularized generative adversarial network (atr-gan) for data augmentation in online process anomaly detection | GAN-augmented classifier  | Time-regularized GAN with augmented filtering and temporal-aware distance constraints.                        | IEEE TASE  | 2021 |
| Regraphgan: A graph generative adversarial network model for dynamic network anomaly detection | Reconstruction error      | Encodes relational dependencies to detect structural graph deviations.                                        | Neural Networks  | 2023 |
| Ganomaly: Semi-supervised anomaly detection via adversarial training | Reconstruction error      | Robust detection of unseen anomalies via joint latent and image space learning.                               | ACCV  | 2018 |
| Ocgan: One-class novelty detection using gans with constrained latent representations | Reconstruction error      | Constrains latent space to suppress outlier reconstruction for one-class anomaly detection.                   | CVPR  | 2019 |
|  f-anogan: Fast unsupervised anomaly detection with generative adversarial networks | Hybrid score (recon&disc) | Latent space mapping for fast anomaly localization in medical imaging.                                        | Medical image analysis  | 2019 |
| Gan ensemble for anomaly detection | Hybrid score (recon&disc) | Uses an ensemble of GANs to enhance detection performance over various base models.                           | AAAI  | 2021 |
|  Anomaly-gan: A data augmentation method for train surface anomaly detection | GAN-augmented classifier  | Mask-guided GAN generates diverse anomalies with local-global discriminators for surface inspection.          | ESWA  | 2023 |
|  Domain adaptive and fine-grained anomaly detection for single-cell sequencing data and beyond | Reconstruction error      | Memory-augmented domain-adaptive GAN for fine-grained anomalous cell detection.                               | IJCAI  | 2024 |
|  Reconstruction-based anomaly detection for multivariate time series using contrastive generative adversarial networks | Reconstruction error      | Integrates contrastive learning and data augmentation into a transformer-based GAN for MTS anomaly detection. |  Information Processing & Management  | 2024 |


#### 1.1.4 Diffusion-based Reconstruction Methods

|  Paper Title       | Technique         | Key Idea         |     Venue     | Year | Ref.  |
|---------------------|---------------------------|---------------------------------------|------------|-----|------|
| Anoddpm: Anomaly detection with denoising diffusion probabilistic models using simplex noise  | Conditional DDPM | Multi-scale simplex noise with partial diffusion to detect anomalies in high-resolution medical images.  | CVPR | 2022 |
|  Glad: towards better reconstruction with global and local adaptive diffusion models for unsupervised anomaly detection | DDIM             | Employs global and local reconstruction with deterministic noise scheduling for enhanced anomaly localization.                             | ECCV  | 2024 |
| R3d-ad: Reconstruction via diffusion for 3d anomaly detection | Conditional DDPM | Latent space regularization and intermediate-step perturbations for precise 3D anomaly detection.                                          | ECCV  | 2024 |
| Diffusionad: Norm-guided one-step denoising diffusion for anomaly detection | DDPM             | Norm-guided single-step refinement with pixel-wise segmentation for improved anomaly localization.                                         | IEEE TPAMI  | 2025 |
|  Dual conditioned motion diffusion for pose-based video anomaly detection | Conditional DDPM | Combines conditioned motion and embedding in a DDPM-based model with temporal-spatial associations for pose-based video anomaly detection. | AAAI  | 2025 |
| Unsupervised surface anomaly detection with diffusion probabilistic model | Conditional DDPM | Noisy conditioning embeddings and interpolation channels to diversify the reconstruction process and improve noise guidance mitigation.    | ICCV  | 2023 |
| A diffusion-based framework for multi-class anomaly detection | Conditional DDPM | Semantic feature fusion of multi-scale features, linking SD and the semantic-guided network, thereby maintaining semantic consistency.     | AAAI  | 2024 |
| Dynamic addition of noise in a diffusion model for anomaly detection | Conditional DDIM | Adaptively modulates the noising process based on anomaly estimates.                                                                       | CVPR  | 2024 |
| Dzad: Diffusion-based zero-shot anomaly detection | Conditional DDPM | Utilizes noise features during the diffusion denoising process to enhance zero-shot generalization.                                        | AAAI  | 2025 |


#### 1.1.5 Hybrid Reconstruction Methods

Beyond standalone AEs, VAEs, GANs, and diffusion models, a growing body of work explores hybrid reconstruction paradigms that aim to combine the strengths of multiple generative models for more effective anomaly detection.
These hybrid approaches seek to improve reconstruction fidelity, capture more expressive latent distributions, and enhance training stability.

|  Paper Title       | Technique         |   Venue     | Year | Ref.  |
|---------------------|---------------------------|---------------------------------------|------------|-----|
| Anomaly detection with adversarial dual autoencoders  | GAN+AE | Arxiv | 2019 |



### 1.2 Representation-driven Approaches

Representation-driven anomaly detection approaches aim to identify anomalies by leveraging discriminative feature representations.
The rise of foundation models, notably LLMs and MLLMs, marks a paradigm shift in AI, delivering unprecedentedly powerful and versatile representation learning capabilities.
Next, we review the representation-driven anomaly detection approaches based on various foundation model types.

#### 1.2.1 LLM-based Methods

| Paper Title      | FM               | Data              | Scenario                                |     Venue     | Year | Ref.  |
|--------------------------|------------------|-------------------|-----------------------------------------|------|------|------|
| Logprompt: Prompt engineering towards zero-shot and interpretable log analysis   | GPT-3.5     | Log       | Software maintenance log     |ICSE-Companion   |  2024 |
|  Face it yourselves: An llm-based two-stage strategy to localize configuration errors via logs | GPT-4            | Log               | Localize configuration errors           | ISSTA  |  2024 |
| Audit-llm: Multi-agent collaboration for log-based insider threat detection | GPT-3.5          | Log               | Log-based insider threat detection      | Arxiv  |  2024 |
|  Logicode: an llm-driven framework for logical anomaly detection | GPT-4            | Text              | Industrial logical text                 | IEEE TASE  |  2024 |
|  Spiced: Syntactical bug and trojan pattern identification in a/ms circuits using llm-enhanced detection | GPT-3.5          | Signal            | Trojan detection in A/MS designs        | PAINE  |  2024 |
|  Real-time anomaly detection and reactive planning with large language models | GPT-4            | Text              | Robotic system monitor                  | Arxiv  |  2024 |
|  Large language models can deliver accurate and interpretable time series anomaly detection | GPT-4            | Time series       | Univariate time series                  | Arxiv  |  2024 |
|  Large language model guided knowledge distillation for time series anomaly detection | GPT-2            | Time series       | Univariate and multivariate time series | Arxiv  |  2024 |
| Can llms serve as time series anomaly detectors? | GPT-4            | Time series       | Univariate time series                  | Arxiv  |  2024 |
| Anomaly detection on unstable logs with gpt models | GPT-3            | Log               | Software systems log                    | Arxiv  |  2024 |
| Large language models can be zero-shot anomaly detectors for time series? | GPT-3.5          | Time series       | Univariate time series                  |  Arxiv  |  2024 |
| Dabl: Detecting semantic anomalies in business processes using large language models | LLaMA-2          | Text              | Business semantic text                  | AAAI  |  2025 |


#### 1.2.2 MLLM-based Methods

| Paper Title      | FM               | Data              | Scenario                                |     Venue     | Year | Ref.  |
|--------------------------|------------------|-------------------|-----------------------------------------|------|------|------|
| Random word data augmentation with clip for zero-shot anomaly detection   | CLIP             | Text & Image      | Industrial image      | Arxiv  | 2023 |
| Myriad: Large multimodal model by applying vision experts for industrial anomaly detection | GPT-3.5          | Text & Image      | Industrial image                        | Arxiv  | 2023 |
| Anomalyclip: Object-agnostic prompt learning for zero-shot anomaly detection | CLIP             | Text & Image      | Industrial and medical image            | Arxiv  | 2023 |
| Winclip: Zero-/few-shot anomaly classification and segmentation | OpenCLIP         | Text & Image      | Industrial image                        | CVPR  | 2023 |
| Anomalygpt: Detecting industrial anomalies using large vision-language models | Vicuna           | Text & Image      | Industrial image                        | AAAI  | 2024 |
| Filo: Zero-shot anomaly detection by fine-grained description and high-quality localization | CLIP             | Text & Image      | Industrial image                        | ACM MM  | 2024 |
| Promptad: Learning prompts with only normal samples for few-shot anomaly detection | CLIP             | Text & Image      | Industrial image                        | CVPR  | 2024 |
| Hawk: Learning to understand open-world video anomalies | LLaMA-2          | Text & Image      | Surveillance image                      | NeurIPS  | 2024 |
| Video anomaly detection and explanation via large language models | LLaMA            | Text & Video      | Surveillance and traffic scene video    | Arxiv  | 2024 |
| Adaclip: Adapting clip with hybrid learnable prompts for zero-shot anomaly detection | CLIP             | Text & Image      | Industrial and medical image            | ECCV  | 2024 |
|  One-to-normal: Anomaly personalization for few-shot anomaly detection | CLIP             | Text & Image      | Industrial, medical, and semantic image | NeurIPS  | 2024 |
| Toward generalist anomaly detection via in-context residual learning with few-shot sample prompts | OpenCLIP         | Text & Image      | Industrial, medical, and semantic image | CVPR  | 2024 |
| Clip-ad: A language-guided staged dual-path model for zero-shot anomaly detection | CLIP             | Text & Image      | Industrial and medical image            | IJCAI  | 2024 |
| Adapting visual-language models for generalizable anomaly detection in medical images | CLIP             | Text & Image      | Medical image                           | CVPR  | 2024 |
| Vera: Explainable video anomaly detection via verbalized learning of vision-language models | InternVL2        | Text & Image      | Surveillance image                      | CVPR  | 2025 |
|  Holmes-vau: Towards long-term video anomaly understanding at any granularity | InternVL2        | Text & Video      | Surveillance video                      | CVPR  | 2025 |
| Univad: A training-free unified model for few-shot visual anomaly detection | CLIP             | Text & Image      | Industrial, logical, and medical image  | CVPR  | 2025 |
| Aa-clip: Enhancing zero-shot anomaly detection via anomaly-aware clip | CLIP             | Text & Image      | Industrial and medical image            | CVPR  | 2025 |
| Bayesian prompt flow learning for zero-shot anomaly detection | CLIP             | Text & Image      | Industrial and medical image            | CVPR  | 2025 |
| Echotraffic: Enhancing traffic anomaly understanding with audio-visual insights | LLaMA-2          | Text, Video&Audio | Traffic anomaly understanding           | CVPR  | 2025 |

#### 1.2.3 Hybrid Representation-driven Methods

Recent studies have explored hybrid pipelines combining LLMs and MLLMs, leveraging their complementary strengths for anomaly detection.

| Paper Title      | FM               | Data              | Scenario                                |     Venue     | Year | Ref.  |
|--------------------------|------------------|-------------------|-----------------------------------------|------|------|------|
| Unsupervised video anomaly detection based on similarity with predefined text descriptions   | ChatGPT & CLIP   | Text & Image      | Surveillance image   |  Sensors | 2023 |
|  Harnessing large language models for training-free video anomaly detection | LLaMA-2 & BLIP-2 | Text & Video      | Surveillance video                      | CVPR  | 2024 |
| Follow the rules: reasoning for video anomaly detection with large language models | GPT-4 & Mistral  | Text & Video      | Surveillance video                      | ECCV  | 2024 |
|  Do llms understand visual anomalies? uncovering llm’s capabilities in zero-shot anomaly detection | GPT-3.5 & CLIP   | Text & Image      | Industrial image                        | ACM MM  | 2024 |


### 1.3 Density Estimation-driven Approaches

Unlike reconstruction-driven methods that identify anomalies based on reconstruction errors, or representation-driven methods that assess deviations within learned embeddings, density estimation-driven approaches rely on the likelihood or probability density as their core anomaly-scoring metric.
The central premise behind density estimation-driven anomaly detection is intuitive and robust: anomalies are inherently associated with low-density regions within the learned data manifold.

#### 1.3.1 Explicit Density-based Methods

|  Paper Title       | Technique         |   Venue     | Year | Ref.  |
|---------------------|---------------------------|---------------------------------------|------------|-----|
| Supervised anomaly detection based on deep autoregressive density estimators  | Autoregressive Models | Arxiv | 2019 |
| Shaken, and stirred: long-range dependencies enable robust outlier detection with pixelcnn++  | Autoregressive Models | IJCAI | 2023 |
|  | Normalizing Flows |   |   |
|  | Energy-Based Models |   |   |


#### 1.3.2 Implicit Density-based Methods

|  Paper Title       | Technique         |   Venue     | Year | Ref.  |
|---------------------|---------------------------|---------------------------------------|------------|-----|
|  | Likelihood-based VAE Variants |   |   |
|    | Density-based Diffusion Methods |   |   |


## 2 Complementary Generative Tasks

Anomaly synthesis and anomaly restoration have emerged as two closely related tasks that extend the scope of generative modeling in anomaly detection.
Anomaly synthesis focuses on creating realistic abnormal instances, enabling data augmentation for training, stress-testing detection models, and constructing more comprehensive evaluation benchmarks.
Anomaly restoration aims to recover anomalous inputs back to their plausible normal states, supporting interpretability, diagnostics, and even operational restoration in domains such as medical and industrial inspection.

### 2.1 Generative Anomaly Synthesis

|  Paper Title       | Method        |   Venue     | Year | Ref.  |
|---------------------|---------------------------|---------------------------------------|------------|-----|
|  | Patch-based Methods |   |   |
|  | Adversarial Methods |   |   |
|    | Conditional Diffusion-based Methods |   |   |

### 2.2 Generative anomaly restoration

|  Paper Title       | Method        |   Venue     | Year | Ref.  |
|---------------------|---------------------------|---------------------------------------|------------|-----|
|  | Attribute-based Methods |   |   |
|  | GAN-inversion-based Methods |   |   |
|    | Diffusion-based Methods |   |   |



## 3 Applications

### 3.1 Industrial manufacturing
...

### 3.2 Medical field
...

### 3.3 Finance market

...

### 3.4 Cybersecurity

...
