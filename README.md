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
| Variational autoencoder-based anomaly detection using reconstruction probability | Utilizes reconstruction probability from VAE as anomaly score    | Special lecture on IE    | 2015 |[[paper]](http://dm.snu.ac.kr/static/docs/TR/SNUDM-TR-2015-03.pdf)
| Generative neural networks for anomaly detection in crowded scenes | Two-stage VAE with GMM for fast filtering and hierarchical feature learning                             | IEEE TIFS | 2018 |[[paper]](https://pure.ulster.ac.uk/files/12666443/S2VAE_manuscript_nobio_20181015_black.pdf)
| Robust anomaly detection for multivariate time series through stochastic recurrent neural network | Stochastic VAE capturing robust MTS representations with variable connection and planar normalizing flow | 	SIGKDD | 2019 |[[paper]](https://netman.aiops.org/wp-content/uploads/2019/08/OmniAnomaly_camera-ready.pdf)
| Video anomaly detection and localization via Gaussian mixture fully convolutional variational autoencoder | Two-stream GMM-VAE learning normal representations for patch-wise scoring                 | CVIU | 2020 |[[paper]](https://arxiv.org/pdf/1805.11223)
| Deep variational graph convolutional recurrent network for multivariate time series anomaly detection | Deep variational graph convolutional recurrent network to model hierarchical spatiotemporal MTS structure. | ICML | 2022 |[[paper]](https://proceedings.mlr.press/v162/chen22x/chen22x.pdf)
| Learning multi-pattern normalities in the frequency domain for efficient time series anomaly detection | Frequency-domain method featuring multi-normal pattern modeling                                       | ICDE | 2024 |[[paper]](https://arxiv.org/pdf/2311.16191)
| Tri-vae: Triplet variational autoencoder for unsupervised anomaly detection in brain tumor mri | Triplet VAE enforcing lesion-normal separation via disentangled metric learning          | CVPR | 2024 |[[paper]](https://openaccess.thecvf.com/content/CVPR2024W/VAND/papers/Wijanarko_Tri-VAE_Triplet_Variational_Autoencoder_for_Unsupervised_Anomaly_Detection_in_Brain_CVPRW_2024_paper.pdf)
|  Unsupervised anomaly localization using variational auto-encoders | Pixel-wise KL-divergence for robust anomaly scoring                                              | MICCAI | 2018 |[[paper]](https://arxiv.org/pdf/1907.02796)
| Anomaly detection of time series with smoothness-inducing sequential variational auto-encoder | Sequential VAE with smoothness-inducing prior for latent temporal consistency    | IEEE TNNLS | 2020 |[[paper]](https://arxiv.org/pdf/2102.01331)
|  Robust outlier detection by de-biasing vae likelihoods | Post hoc likelihood debiasing via bias correction and contrast normalization        | CVPR | 2022 |[[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Chauhan_Robust_Outlier_Detection_by_De-Biasing_VAE_Likelihoods_CVPR_2022_paper.pdf)
|  Anomaly detection in time series with robust variational quasi-recurrent autoencoders | Quasi-recurrent VAE with robust α-, β-, and γ-divergence               | ICDE | 2022 |[[paper]](https://ieeexplore.ieee.org/abstract/document/9835268/)
| Vae-based deep svdd for anomaly detection | Joint VAE-SVDD learning in latent hypersphere                     | Neurocomputing | 2020 |[[paper]](https://www.sciencedirect.com/science/article/pii/S0925231221006470)
|  A semi-supervised vae based active anomaly detection framework in multivariate time series for online systems | Semi-supervised VAE updated via uncertainty-guided active learning     | WWW | 2022 |[[paper]](https://scholar.archive.org/work/gcpuysh4yrd7xoukmrdetn4swq/access/wayback/https://dl.acm.org/doi/pdf/10.1145/3485447.3511984)
| Situation-aware multivariate time series anomaly detection through active learning and contrast vae-based models in large distributed systems | Contrastive VAE with periodic joint training and query model    | IEEE JSAC | 2022 |[[paper]](https://ieeexplore.ieee.org/abstract/document/9844802)
| Unsupervised anomaly detection on microservice traces through graph vae | Dual-variable VAE with dispatch learning of structure and time features         | WWW | 2024 |[[paper]](https://dl.acm.org/doi/abs/10.1145/3543507.3583215)
| Vaeat: Variational autoeencoder with adversarial training for multivariate time series anomaly detection | Adversarial VAE with attention and dual-decoder regularization        |  Information Sciences | 2024 |[[paper]](https://www.sciencedirect.com/science/article/pii/S0020025524007667)


#### 1.1.3 GAN-based Reconstruction Methods

|   Paper Title   | Anomaly Score             | Key Idea                                 |     Venue     | Year | Ref.  |
|---------------------|---------------------------|----------------------------------------------|------------|------|----|
| Unsupervised anomaly detection with generative adversarial networks to guide marker discovery  | Hybrid score (recon&disc) | Early GAN-based method combining visual fidelity and discriminative feedback | IPMI  | 2017 |[[paper]](https://arxiv.org/abs/1703.05921)
| Mad-gan: Multivariate anomaly detection for time series data with generative adversarial networks | Hybrid score (recon&disc) | Ensemble discriminator scoring for robust time-series anomaly detection               | ICANN | 2019 |[[paper]](https://link.springer.com/chapter/10.1007/978-3-030-30490-4_56)
| Generative adversarial active learning for unsupervised outlier detection | Discriminator confidence  | Multi-generator strategy for outlier synthesis and a discriminator for detection.                             | IEEE TKDE  | 2019 |[[paper]](https://ieeexplore.ieee.org/abstract/document/8668550)
| Augmented time regularized generative adversarial network (atr-gan) for data augmentation in online process anomaly detection | GAN-augmented classifier  | Time-regularized GAN with augmented filtering and temporal-aware distance constraints.                        | IEEE TASE  | 2021 |[[paper]](https://ieeexplore.ieee.org/abstract/document/9592834)
| Regraphgan: A graph generative adversarial network model for dynamic network anomaly detection | Reconstruction error      | Encodes relational dependencies to detect structural graph deviations.                                        | Neural Networks  | 2023 |[[paper]](https://www.sciencedirect.com/science/article/pii/S0893608023003842)
| Ganomaly: Semi-supervised anomaly detection via adversarial training | Reconstruction error      | Robust detection of unseen anomalies via joint latent and image space learning.                               | ACCV  | 2018 |[[paper]](https://link.springer.com/chapter/10.1007/978-3-030-20893-6_39)
| Ocgan: One-class novelty detection using gans with constrained latent representations | Reconstruction error      | Constrains latent space to suppress outlier reconstruction for one-class anomaly detection.                   | CVPR  | 2019 |[[paper]](https://openaccess.thecvf.com/content_CVPR_2019/html/Perera_OCGAN_One-Class_Novelty_Detection_Using_GANs_With_Constrained_Latent_Representations_CVPR_2019_paper.html)
|  f-anogan: Fast unsupervised anomaly detection with generative adversarial networks | Hybrid score (recon&disc) | Latent space mapping for fast anomaly localization in medical imaging.                                        | Medical image analysis  | 2019 |[[paper]](https://www.sciencedirect.com/science/article/pii/S1361841518302640)
| Gan ensemble for anomaly detection | Hybrid score (recon&disc) | Uses an ensemble of GANs to enhance detection performance over various base models.                           | AAAI  | 2021 |[[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/16530)
|  Anomaly-gan: A data augmentation method for train surface anomaly detection | GAN-augmented classifier  | Mask-guided GAN generates diverse anomalies with local-global discriminators for surface inspection.          | ESWA  | 2023 |[[paper]](https://www.sciencedirect.com/science/article/pii/S0957417423007868)
|  Domain adaptive and fine-grained anomaly detection for single-cell sequencing data and beyond | Reconstruction error      | Memory-augmented domain-adaptive GAN for fine-grained anomalous cell detection.                               | IJCAI  | 2024 |[[paper]](https://arxiv.org/abs/2404.17454)
|  Reconstruction-based anomaly detection for multivariate time series using contrastive generative adversarial networks | Reconstruction error      | Integrates contrastive learning and data augmentation into a transformer-based GAN for MTS anomaly detection. |  Information Processing & Management  | 2024 |[[paper]](https://www.sciencedirect.com/science/article/pii/S0306457323003060)


#### 1.1.4 Diffusion-based Reconstruction Methods

|  Paper Title       | Technique         | Key Idea         |     Venue     | Year | Ref.  |
|---------------------|---------------------------|---------------------------------------|------------|-----|------|
| Anoddpm: Anomaly detection with denoising diffusion probabilistic models using simplex noise  | Conditional DDPM | Multi-scale simplex noise with partial diffusion to detect anomalies in high-resolution medical images.  | CVPR | 2022 |[[paper]](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/html/Wyatt_AnoDDPM_Anomaly_Detection_With_Denoising_Diffusion_Probabilistic_Models_Using_Simplex_CVPRW_2022_paper.html)
|  Glad: towards better reconstruction with global and local adaptive diffusion models for unsupervised anomaly detection | DDIM             | Employs global and local reconstruction with deterministic noise scheduling for enhanced anomaly localization.                             | ECCV  | 2024 |[[paper]](https://link.springer.com/chapter/10.1007/978-3-031-73209-6_1)
| R3d-ad: Reconstruction via diffusion for 3d anomaly detection | Conditional DDPM | Latent space regularization and intermediate-step perturbations for precise 3D anomaly detection.                                          | ECCV  | 2024 |[[paper]](https://link.springer.com/chapter/10.1007/978-3-031-72764-1_6)
| Diffusionad: Norm-guided one-step denoising diffusion for anomaly detection | DDPM             | Norm-guided single-step refinement with pixel-wise segmentation for improved anomaly localization.                                         | IEEE TPAMI  | 2025 |[[paper]](https://ieeexplore.ieee.org/abstract/document/11005495)
|  Dual conditioned motion diffusion for pose-based video anomaly detection | Conditional DDPM | Combines conditioned motion and embedding in a DDPM-based model with temporal-spatial associations for pose-based video anomaly detection. | AAAI  | 2025 |[[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32829)
| Unsupervised surface anomaly detection with diffusion probabilistic model | Conditional DDPM | Noisy conditioning embeddings and interpolation channels to diversify the reconstruction process and improve noise guidance mitigation.    | ICCV  | 2023 |[[paper]](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Unsupervised_Surface_Anomaly_Detection_with_Diffusion_Probabilistic_Model_ICCV_2023_paper.html)
| A diffusion-based framework for multi-class anomaly detection | Conditional DDPM | Semantic feature fusion of multi-scale features, linking SD and the semantic-guided network, thereby maintaining semantic consistency.     | AAAI  | 2024 |[[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/28690)
| Dynamic addition of noise in a diffusion model for anomaly detection | Conditional DDIM | Adaptively modulates the noising process based on anomaly estimates.                                                                       | CVPR  | 2024 |[[paper]](https://openaccess.thecvf.com/content/CVPR2024W/VAND/html/Tebbe_Dynamic_Addition_of_Noise_in_a_Diffusion_Model_for_Anomaly_CVPRW_2024_paper.html)
| Dzad: Diffusion-based zero-shot anomaly detection | Conditional DDPM | Utilizes noise features during the diffusion denoising process to enhance zero-shot generalization.                                        | AAAI  | 2025 |[[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/33099)


#### 1.1.5 Hybrid Reconstruction Methods

Beyond standalone AEs, VAEs, GANs, and diffusion models, a growing body of work explores hybrid reconstruction paradigms that aim to combine the strengths of multiple generative models for more effective anomaly detection.
These hybrid approaches seek to improve reconstruction fidelity, capture more expressive latent distributions, and enhance training stability.

|  Paper Title       | Technique         |   Venue     | Year | Ref.  |
|---------------------|---------------------------|---------------------------------------|------------|-----|
| Anomaly detection with adversarial dual autoencoders  | GAN+AE | Arxiv | 2019 |[[paper]](https://arxiv.org/abs/1902.06924)
| Autoencoding beyond pixels using a learned similarity metric  | GAN+VAE | ICML | 2016 |[[paper]](https://proceedings.mlr.press/v48/larsen16)
| Cloud-gan: Cloud generation adversarial networks for anomaly detection  | GAN+VAE | Pattern Recognition | 2025 |[[paper]](https://www.sciencedirect.com/science/article/pii/S0031320324006174)
| Anomaly detection with conditioned denoising diffusion models  | Diffusion model+AE | DAGM GCPR | 2024 |[[paper]](https://link.springer.com/chapter/10.1007/978-3-031-85181-0_12)
| Diffgad: A diffusion-based unsupervised graph anomaly detector  | Diffusion model+AE | ICLR | 2025 |[[paper]](https://arxiv.org/abs/2410.06549)
| A unified latent schrodinger bridge diffusion model for unsupervised anomaly detection and localization  | Diffusion model+VAE | CVPR | 2025 |[[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Akshay_A_Unified_Latent_Schrodinger_Bridge_Diffusion_Model_for_Unsupervised_Anomaly_CVPR_2025_paper.html)



### 1.2 Representation-driven Approaches

Representation-driven anomaly detection approaches aim to identify anomalies by leveraging discriminative feature representations.
The rise of foundation models, notably LLMs and MLLMs, marks a paradigm shift in AI, delivering unprecedentedly powerful and versatile representation learning capabilities.
Next, we review the representation-driven anomaly detection approaches based on various foundation model types.

#### 1.2.1 LLM-based Methods

| Paper Title      | FM               | Data              | Scenario                                |     Venue     | Year | Ref.  |
|--------------------------|------------------|-------------------|-----------------------------------------|------|------|------|
| Logprompt: Prompt engineering towards zero-shot and interpretable log analysis   | GPT-3.5     | Log       | Software maintenance log     |ICSE-Companion   |  2024 |[[paper]](https://dl.acm.org/doi/abs/10.1145/3639478.3643108)
|  Face it yourselves: An llm-based two-stage strategy to localize configuration errors via logs | GPT-4            | Log               | Localize configuration errors           | ISSTA  |  2024 |[[paper]](https://dl.acm.org/doi/abs/10.1145/3650212.3652106)
| Audit-llm: Multi-agent collaboration for log-based insider threat detection | GPT-3.5          | Log               | Log-based insider threat detection      | Arxiv  |  2024 |[[paper]](https://arxiv.org/abs/2408.08902)
|  Logicode: an llm-driven framework for logical anomaly detection | GPT-4            | Text              | Industrial logical text                 | IEEE TASE  |  2024 |[[paper]](https://ieeexplore.ieee.org/abstract/document/10710633)
|  Spiced: Syntactical bug and trojan pattern identification in a/ms circuits using llm-enhanced detection | GPT-3.5          | Signal            | Trojan detection in A/MS designs        | PAINE  |  2024 |[[paper]](https://ieeexplore.ieee.org/abstract/document/10792717)
|  Real-time anomaly detection and reactive planning with large language models | GPT-4            | Text              | Robotic system monitor                  | Arxiv  |  2024 |[[paper]](https://arxiv.org/abs/2407.08735)
|  Large language models can deliver accurate and interpretable time series anomaly detection | GPT-4            | Time series       | Univariate time series                  | Arxiv  |  2024 |[[paper]](https://arxiv.org/abs/2405.15370)
|  Large language model guided knowledge distillation for time series anomaly detection | GPT-2            | Time series       | Univariate and multivariate time series | Arxiv  |  2024 |[[paper]](https://arxiv.org/abs/2401.15123)
| Can llms serve as time series anomaly detectors? | GPT-4            | Time series       | Univariate time series                  | Arxiv  |  2024 |[[paper]](https://arxiv.org/abs/2408.03475)
| Anomaly detection on unstable logs with gpt models | GPT-3            | Log               | Software systems log                    | Arxiv  |  2024 |[[paper]](https://ui.adsabs.harvard.edu/abs/2024arXiv240607467H/abstract)
| Large language models can be zero-shot anomaly detectors for time series? | GPT-3.5          | Time series       | Univariate time series                  |  Arxiv  |  2024 |[[paper]](https://arxiv.org/abs/2405.14755)
| Dabl: Detecting semantic anomalies in business processes using large language models | LLaMA-2          | Text              | Business semantic text                  | AAAI  |  2025 |[[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/33277)


#### 1.2.2 MLLM-based Methods

| Paper Title      | FM               | Data              | Scenario                                |     Venue     | Year | Ref.  |
|--------------------------|------------------|-------------------|-----------------------------------------|------|------|------|
| Random word data augmentation with clip for zero-shot anomaly detection   | CLIP             | Text & Image      | Industrial image      | Arxiv  | 2023 |[[paper]](https://arxiv.org/abs/2308.11119)
| Myriad: Large multimodal model by applying vision experts for industrial anomaly detection | GPT-3.5          | Text & Image      | Industrial image                        | Arxiv  | 2023 |[[paper]](https://arxiv.org/abs/2310.19070)
| Anomalyclip: Object-agnostic prompt learning for zero-shot anomaly detection | CLIP             | Text & Image      | Industrial and medical image            | Arxiv  | 2023 |[[paper]](https://arxiv.org/abs/2310.18961)
| Winclip: Zero-/few-shot anomaly classification and segmentation | OpenCLIP         | Text & Image      | Industrial image                        | CVPR  | 2023 |[[paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.html)
| Anomalygpt: Detecting industrial anomalies using large vision-language models | Vicuna           | Text & Image      | Industrial image                        | AAAI  | 2024 |[[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/27963)
| Filo: Zero-shot anomaly detection by fine-grained description and high-quality localization | CLIP             | Text & Image      | Industrial image                        | ACM MM  | 2024 |[[paper]](https://dl.acm.org/doi/abs/10.1145/3664647.3680685)
| Promptad: Learning prompts with only normal samples for few-shot anomaly detection | CLIP             | Text & Image      | Industrial image                        | CVPR  | 2024 |[[paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Li_PromptAD_Learning_Prompts_with_only_Normal_Samples_for_Few-Shot_Anomaly_CVPR_2024_paper.html)
| Hawk: Learning to understand open-world video anomalies | LLaMA-2          | Text & Image      | Surveillance image                      | NeurIPS  | 2024 |[[paper]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fca83589e85cb061631b7ebc5db5d6bd-Abstract-Conference.html)
| Video anomaly detection and explanation via large language models | LLaMA            | Text & Video      | Surveillance and traffic scene video    | Arxiv  | 2024 |[[paper]](https://arxiv.org/abs/2401.05702)
| Adaclip: Adapting clip with hybrid learnable prompts for zero-shot anomaly detection | CLIP             | Text & Image      | Industrial and medical image            | ECCV  | 2024 |[[paper]](https://link.springer.com/chapter/10.1007/978-3-031-72761-0_4)
|  One-to-normal: Anomaly personalization for few-shot anomaly detection | CLIP             | Text & Image      | Industrial, medical, and semantic image | NeurIPS  | 2024 |[[paper]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8f4477b086a9c97e30d1a0621ea6b2f5-Abstract-Conference.html)
| Toward generalist anomaly detection via in-context residual learning with few-shot sample prompts | OpenCLIP         | Text & Image      | Industrial, medical, and semantic image | CVPR  | 2024 |[[paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Zhu_Toward_Generalist_Anomaly_Detection_via_In-context_Residual_Learning_with_Few-shot_CVPR_2024_paper.html)
| Clip-ad: A language-guided staged dual-path model for zero-shot anomaly detection | CLIP             | Text & Image      | Industrial and medical image            | IJCAI  | 2024 |[[paper]](https://link.springer.com/chapter/10.1007/978-981-97-9003-6_2)
| Adapting visual-language models for generalizable anomaly detection in medical images | CLIP             | Text & Image      | Medical image                           | CVPR  | 2024 |[[paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_Adapting_Visual-Language_Models_for_Generalizable_Anomaly_Detection_in_Medical_Images_CVPR_2024_paper.html)
| Vera: Explainable video anomaly detection via verbalized learning of vision-language models | InternVL2        | Text & Image      | Surveillance image                      | CVPR  | 2025 |[[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Ye_VERA_Explainable_Video_Anomaly_Detection_via_Verbalized_Learning_of_Vision-Language_CVPR_2025_paper.html)
|  Holmes-vau: Towards long-term video anomaly understanding at any granularity | InternVL2        | Text & Video      | Surveillance video                      | CVPR  | 2025 |[[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Holmes-VAU_Towards_Long-term_Video_Anomaly_Understanding_at_Any_Granularity_CVPR_2025_paper.html)
| Univad: A training-free unified model for few-shot visual anomaly detection | CLIP             | Text & Image      | Industrial, logical, and medical image  | CVPR  | 2025 |[[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Gu_UniVAD_A_Training-free_Unified_Model_for_Few-shot_Visual_Anomaly_Detection_CVPR_2025_paper.html)
| Aa-clip: Enhancing zero-shot anomaly detection via anomaly-aware clip | CLIP             | Text & Image      | Industrial and medical image            | CVPR  | 2025 |[[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Ma_AA-CLIP_Enhancing_Zero-Shot_Anomaly_Detection_via_Anomaly-Aware_CLIP_CVPR_2025_paper.html)
| Bayesian prompt flow learning for zero-shot anomaly detection | CLIP             | Text & Image      | Industrial and medical image            | CVPR  | 2025 |[[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Qu_Bayesian_Prompt_Flow_Learning_for_Zero-Shot_Anomaly_Detection_CVPR_2025_paper.html)
| Echotraffic: Enhancing traffic anomaly understanding with audio-visual insights | LLaMA-2          | Text, Video&Audio | Traffic anomaly understanding           | CVPR  | 2025 |[[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Xing_EchoTraffic_Enhancing_Traffic_Anomaly_Understanding_with_Audio-Visual_Insights_CVPR_2025_paper.html)

#### 1.2.3 Hybrid Representation-driven Methods

Recent studies have explored hybrid pipelines combining LLMs and MLLMs, leveraging their complementary strengths for anomaly detection.

| Paper Title      | FM               | Data              | Scenario                                |     Venue     | Year | Ref.  |
|--------------------------|------------------|-------------------|-----------------------------------------|------|------|------|
| Unsupervised video anomaly detection based on similarity with predefined text descriptions   | ChatGPT & CLIP   | Text & Image      | Surveillance image   |  Sensors | 2023 |[[paper]](https://www.mdpi.com/1424-8220/23/14/6256)
|  Harnessing large language models for training-free video anomaly detection | LLaMA-2 & BLIP-2 | Text & Video      | Surveillance video                      | CVPR  | 2024 |[[paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Zanella_Harnessing_Large_Language_Models_for_Training-free_Video_Anomaly_Detection_CVPR_2024_paper.html)
| Follow the rules: reasoning for video anomaly detection with large language models | GPT-4 & Mistral  | Text & Video      | Surveillance video                      | ECCV  | 2024 |[[paper]](https://link.springer.com/chapter/10.1007/978-3-031-73004-7_18)
|  Do llms understand visual anomalies? uncovering llm’s capabilities in zero-shot anomaly detection | GPT-3.5 & CLIP   | Text & Image      | Industrial image                        | ACM MM  | 2024 |[[paper]](https://dl.acm.org/doi/abs/10.1145/3664647.3681190)


### 1.3 Density Estimation-driven Approaches

Unlike reconstruction-driven methods that identify anomalies based on reconstruction errors, or representation-driven methods that assess deviations within learned embeddings, density estimation-driven approaches rely on the likelihood or probability density as their core anomaly-scoring metric.
The central premise behind density estimation-driven anomaly detection is intuitive and robust: anomalies are inherently associated with low-density regions within the learned data manifold.

#### 1.3.1 Explicit Density-based Methods

|  Paper Title       | Technique         |   Venue     | Year | Ref.  |
|---------------------|---------------------------|---------------------------------------|------------|-----|
| Supervised anomaly detection based on deep autoregressive density estimators  | Autoregressive Models | Arxiv | 2019 |[[paper]](https://arxiv.org/abs/1904.06034)
| Shaken, and stirred: long-range dependencies enable robust outlier detection with pixelcnn++  | Autoregressive Models | IJCAI | 2023 |[[paper]](https://research.google/pubs/shaken-and-stirred-long-range-dependencies-enable-robust-outlier-detection-with-pixelcnn/)
| Autoregressive denoising score matching is a good video anomaly detector  | Autoregressive Models | Arxiv | 2025 |[[paper]](https://arxiv.org/abs/2506.23282)
| Anomaly detection for symbolic sequences and time series data  | Autoregressive Models | University of Minnesota | 2009 |[[paper]](https://www.proquest.com/openview/59deed3486e0112b8ad59520b720d764/1?pq-origsite=gscholar&cbl=18750)
| Demand forecasting in smart grid using long short-term memory  | Autoregressive Models | ACMI | 2021 |[[paper]](https://ieeexplore.ieee.org/abstract/document/9528277)
| Varad: Lightweight high-resolution image anomaly detection via visual autoregressive modeling  | Autoregressive Models | IEEE TII | 2025 |[[paper]](https://ieeexplore.ieee.org/abstract/document/10843956)
| Autoregressive density estimation transformers for multivariate time series anomaly detection  | Autoregressive Models | ICASSP | 2025 |[[paper]](https://ieeexplore.ieee.org/abstract/document/10888728)
| Enhancing structural anomaly detection using a bounded autoregressive component  | Autoregressive Models | MSSP | 2024 |[[paper]](https://www.sciencedirect.com/science/article/pii/S0888327024001778)
| Latent space autoregression for novelty detection  | Autoregressive Models | CVPR | 2019 |[[paper]](https://openaccess.thecvf.com/content_CVPR_2019/html/Abati_Latent_Space_Autoregression_for_Novelty_Detection_CVPR_2019_paper.html)
| Intrusion detection for high-speed railways based on unsupervised anomaly detection models  | Autoregressive Models | Applied Intelligence | 2023 |[[paper]](https://link.springer.com/article/10.1007/s10489-022-03911-8)
| Varade: a variational-based autoregressive model for anomaly detection on the edge  | Autoregressive Models | DAC | 2024 |[[paper]](https://dl.acm.org/doi/abs/10.1145/3649329.3655691)
| Pixel recurrent neural networks  | Autoregressive Models | ICML | 2016 |[[paper]](https://proceedings.mlr.press/v48/oord16.html)
| Anomaly detection in automobile control network data with long short-term memory networks  | Autoregressive Models | DSAA | 2016 |[[paper]](https://ieeexplore.ieee.org/abstract/document/7796898)
| Anomaly detection in raw audio using deep autoregressive networks  | Autoregressive Models | ICASSP | 2019 |[[paper]](https://ieeexplore.ieee.org/abstract/document/8683414)
| Lstm learning with bayesian and gaussian processing for anomaly detection in industrial iot  | Autoregressive Models | IEEE TII | 2019 |[[paper]](https://ieeexplore.ieee.org/abstract/document/8896029)
| Deep learning-based anomaly detection for connected autonomous vehicles using spatiotemporal information  | Autoregressive Models | IEEE T-ITS | 2023 |[[paper]](https://ieeexplore.ieee.org/abstract/document/10164214)
| Gru-based interpretable multivariate time series anomaly detection in industrial control system  | Autoregressive Models | Computers & Security | 2023 |[[paper]](https://www.sciencedirect.com/science/article/pii/S0167404823000044)
| Pixelcnn++: Improving the pixelcnn with discretized logistic mixture likelihood and other modifications  | Autoregressive Models | ICLR | 2017 |[[paper]](https://arxiv.org/abs/1701.05517)
| Pixeldefend: Leveraging generative models to understand and defend against adversarial examples  | Autoregressive Models | Arxiv | 2017 |[[paper]](https://arxiv.org/abs/1710.10766)
| Unsupervised brain anomaly detection and segmentation with transformers  | Autoregressive Models | MIDL | 2021 |[[paper]](https://arxiv.org/abs/2102.11650)
| Self-supervised video forensics by audio-visual anomaly detection  | Autoregressive Models | CVPR | 2023 |[[paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Feng_Self-Supervised_Video_Forensics_by_Audio-Visual_Anomaly_Detection_CVPR_2023_paper.html)
| Loggpt: Log anomaly detection via gpt  | Autoregressive Models | BigData | 2023 |[[paper]](https://ieeexplore.ieee.org/abstract/document/10386543)
| Early exploration of using chatgpt for log-based anomaly detection on parallel file systems logs  | Autoregressive Models | HPDC | 2023 |[[paper]](https://dl.acm.org/doi/abs/10.1145/3588195.3595943)
| Improving log-based anomaly detection by pre-training hierarchical transformers  | Autoregressive Models | IEEE TC | 2023 |[[paper]](https://ieeexplore.ieee.org/abstract/document/10070784)
| Gelog: a gpt-enhanced log representation method for anomaly detection  | Autoregressive Models | ICPC | 2025 |[[paper]](https://www.computer.org/csdl/proceedings-article/icpc/2025/022300a524/27CwQU7IYFi)
| Unleash gpt-2 power for event detection  | Autoregressive Models | ACL-IJCNLP | 2021 |[[paper]](https://aclanthology.org/2021.acl-long.490/)
| Supervised abnormal event detection based on chatgpt attention mechanism  | Autoregressive Models | MTA | 2024 |[[paper]](https://link.springer.com/article/10.1007/s11042-024-18551-y)
| Detecting spacecraft anomalies using lstms and nonparametric dynamic thresholding  | Autoregressive Models | KDD | 2018 |[[paper]](https://dl.acm.org/doi/abs/10.1145/3219819.3219845)
| Unsupervised anomaly detection with lstm neural networks  | Autoregressive Models | IEEE TNNLS | 2019 |[[paper]](https://ieeexplore.ieee.org/abstract/document/8836638)
| Anomaly detection through latent space restoration using vector quantized variational autoencoders  | Autoregressive Models | ISBI | 2021 |[[paper]](https://ieeexplore.ieee.org/abstract/document/9433778)
| Why normalizing flows fail to detect out-of-distribution data | Normalizing Flows | NeurIPS  | 2020  |[[paper]](https://proceedings.neurips.cc/paper/2020/hash/ecb9fe2fbb99c31f567e9823e884dbec-Abstract.html)
| Same same but differnet: Semi-supervised defect detection with normalizing flows | Normalizing Flows | WACV  | 2021  |[[paper]](https://openaccess.thecvf.com/content/WACV2021/html/Rudolph_Same_Same_but_DifferNet_Semi-Supervised_Defect_Detection_With_Normalizing_Flows_WACV_2021_paper.html)
| Cflow-ad: Real-time unsupervised anomaly detection with localization via conditional normalizing flows | Normalizing Flows | WACV  | 2022  |[[paper]](https://openaccess.thecvf.com/content/WACV2022/html/Gudovskiy_CFLOW-AD_Real-Time_Unsupervised_Anomaly_Detection_With_Localization_via_Conditional_Normalizing_WACV_2022_paper.html?ref=https://githubhelp.com)
| Fully convolutional cross-scale-flows for image-based defect detection | Normalizing Flows | WACV  | 2022  |[[paper]](https://openaccess.thecvf.com/content/WACV2022/html/Rudolph_Fully_Convolutional_Cross-Scale-Flows_for_Image-Based_Defect_Detection_WACV_2022_paper.html)
| Msflow: Multiscale flow-based framework for unsupervised anomaly detection | Normalizing Flows | IEEE TNNLS  | 2024  |[[paper]](https://ieeexplore.ieee.org/abstract/document/10384766)
| Self-supervised normalizing flows for image anomaly detection and localization | Normalizing Flows | CVPR  | 2023  |[[paper]](https://openaccess.thecvf.com/content/CVPR2023W/VAND/html/Chiu_Self-Supervised_Normalizing_Flows_for_Image_Anomaly_Detection_and_Localization_CVPRW_2023_paper.html)
| Fastflow: Unsupervised anomaly detection and localization via 2d normalizing flows | Normalizing Flows | Arxiv  | 2021  |[[paper]](https://arxiv.org/abs/2111.07677)
| Hierarchical gaussian mixture normalizing flow modeling for unified anomaly detection | Normalizing Flows | ECCV  | 2024  |[[paper]](https://link.springer.com/chapter/10.1007/978-3-031-73411-3_6)
| Graph-augmented normalizing flows for anomaly detection of multiple time series | Normalizing Flows | ICLR  | 2022  |[[paper]](https://arxiv.org/abs/2202.07857)
| Detecting multivariate time series anomalies with zero known label | Normalizing Flows | AAAI  | 2023  |[[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/25623)
| Label-free multivariate time series anomaly detection | Normalizing Flows | IEEE TKDE  | 2024  |[[paper]](https://ieeexplore.ieee.org/abstract/document/10380724)
| Spatial-temporal graph conditionalized normalizing flows for nuclear power plant multivariate anomaly detection | Normalizing Flows | IEEE TII  | 2024  |[[paper]](https://ieeexplore.ieee.org/abstract/document/10616382)
| Unsupervised video anomaly detection via normalizing flows with implicit latent features | Normalizing Flows | Pattern Recognition  | 2022  |[[paper]](https://www.sciencedirect.com/science/article/pii/S0031320322001844)
| Normalizing flows for human pose anomaly detection | Normalizing Flows | ICCV  | 2023  |[[paper]](https://openaccess.thecvf.com/content/ICCV2023/html/Hirschorn_Normalizing_Flows_for_Human_Pose_Anomaly_Detection_ICCV_2023_paper.html)
| Da-flow: dual attention normalizing flow for skeleton-based video anomaly detection | Normalizing Flows | Arxiv  | 2024  |[[paper]](https://arxiv.org/abs/2406.02976)
| Deep structured energy based models for anomaly detection | Energy-Based Models | ICML  | 2016  |[[paper]](https://proceedings.mlr.press/v48/zhai16.html)
| Self-supervised likelihood estimation with energy guidance for anomaly segmentation in urban scenes | Energy-Based Models | AAAI  | 2024  |[[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/30162)
| Energy-based anomaly detection a new perspective for predicting software failures | Energy-Based Models | ICSE-NIER  | 2019  |[[paper]](https://ieeexplore.ieee.org/abstract/document/8805736)
| Your classifier is secretly an energy based model and you should treat it like one | Energy-Based Models | Arxiv  | 2019  |[[paper]](https://arxiv.org/abs/1912.03263)
| Energy-based out-of-distribution detection | Energy-Based Models | NeurIPS  | 2020  |[[paper]](https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html)
| Elsa: Energy-based learning for semi-supervised anomaly detection | Energy-Based Models | Arxiv  | 2021  |[[paper]](https://arxiv.org/abs/2103.15296)
| Pixel-wise energy-biased abstention learning for anomaly segmentation on complex urban driving scenes | Energy-Based Models | ECCV  | 2022  |[[paper]](https://link.springer.com/chapter/10.1007/978-3-031-19842-7_15)
| Energy-based out-of-distribution detection for graph neural networks | Energy-Based Models | ICLR  | 2023  |[[paper]](https://arxiv.org/abs/2302.02914)
| Uniform: Towards unified framework for anomaly detection on graphs | Energy-Based Models | AAAI  | 2025  |[[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/33369)
| Iterative energy-based projection on a normal data manifold for anomaly localization | Energy-Based Models | ICLR  | 2020  |[[paper]](https://arxiv.org/abs/2002.03734)
| Maximum entropy inverse reinforcement learning of diffusion models with energy-based models | Energy-Based Models | NeurIPS  | 2024  |[[paper]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2bed6c14cd5ea97a9bc1e6094941bde7-Abstract-Conference.html)


#### 1.3.2 Implicit Density-based Methods

|  Paper Title       | Technique         |   Venue     | Year | Ref.  |
|---------------------|---------------------------|---------------------------------------|------------|-----|
| beta-vae: Learning basic visual concepts with a constrained variational framework | Likelihood-based VAE Variants | ICLR  | 2017  |[[paper]](https://openreview.net/forum?id=Sy2fzU9gl)
| Out-of-distribution detection in multi-label datasets using latent space of β-vae | Likelihood-based VAE Variants | SPW  | 2020  |[[paper]](https://ieeexplore.ieee.org/abstract/document/9283847/)
| Efficient out-of-distribution detection using latent space of β-vae for cyber-physical systems | Likelihood-based VAE Variants | ACM TCPS  | 2022  |[[paper]](https://dl.acm.org/doi/full/10.1145/3491243)
| Disentangled anomaly detection for multivariate time series | Likelihood-based VAE Variants | WWW  | 2024  |[[paper]](https://dl.acm.org/doi/abs/10.1145/3589335.3651492)
| Ladder variational autoencoders | Likelihood-based VAE Variants | NeurIPS  | 2016  |[[paper]](https://proceedings.neurips.cc/paper/2016/hash/6ae07dcb33ec3b7c814df797cbda0f87-Abstract.html)
| Biva: A very deep hierarchy of latent variables for generative modeling | Likelihood-based VAE Variants | NeurIPS  | 2019  |[[paper]](https://proceedings.neurips.cc/paper/2019/hash/9bdb8b1faffa4b3d41779bb495d79fb9-Abstract.html)
| Switching gaussian mixture variational rnn for anomaly detection of diverse cdn websites | Likelihood-based VAE Variants | IEEE INFOCOM  | 2022  |[[paper]](https://ieeexplore.ieee.org/abstract/document/9796836)
| Cvad: A generic medical anomaly detector based on cascade vae | Likelihood-based VAE Variants | Arxiv  | 2021  |[[paper]](https://arxiv.org/abs/2110.15811)
| Out-of-distribution detection with an adaptive likelihood ratio on informative hierarchical vae | Likelihood-based VAE Variants | NeurIPS  | 2022  |[[paper]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3066f60a91d652f4dc690637ac3a2f8c-Abstract-Conference.html)
| Importance weighted autoencoders | Likelihood-based VAE Variants | Arxiv  | 2015  |[[paper]](https://arxiv.org/abs/1509.00519)
| Odim: Outlier detection via likelihood of under-fitted generative models | Likelihood-based VAE Variants | ICML  | 2024  |[[paper]](https://arxiv.org/abs/2301.04257)
| Neural network-based score estimation in diffusion models: Optimization and generalization | Density-based Diffusion Methods | ICLR  | 2024  |[[paper]](https://arxiv.org/abs/2401.15604)
| Score approximation, estimation and distribution recovery of diffusion models on low-dimensional data | Density-based Diffusion Methods | ICML  | 2023  |[[paper]](https://proceedings.mlr.press/v202/chen23o.html)
| Maximum likelihood training of score-based diffusion models | Density-based Diffusion Methods | NeurIPS  | 2021  |[[paper]](https://proceedings.neurips.cc/paper/2021/hash/0a9fdbb17feb6ccb7ec405cfb85222c4-Abstract.html)
| Low-rank characteristic tensor density estimation part ii: Compression and latent density estimation | Density-based Diffusion Methods | IEEE TSP  | 2022  |[[paper]](https://ieeexplore.ieee.org/abstract/document/9740538)
| Your diffusion model secretly knows the dimension of the data manifold | Density-based Diffusion Methods | Arxiv  | 2022  |[[paper]](https://arxiv.org/abs/2212.12611)
| Disyre: Diffusion-inspired synthetic restoration for unsupervised anomaly detection | Density-based Diffusion Methods | ISBI  | 2024  |[[paper]](https://ieeexplore.ieee.org/abstract/document/10635161)
| D3ad: Dynamic denoising diffusion probabilistic model for anomaly detection | Density-based Diffusion Methods | ICLR  | 2024  |[[paper]](https://openreview.net/forum?id=7jUQHmz4Tq)
| Closing the ode–sde gap in score-based diffusion models through the fokker–planck equation | Density-based Diffusion Methods | Philosophical Transactions A  | 2025  |[[paper]](https://royalsocietypublishing.org/doi/full/10.1098/rsta.2024.0503)
| Adbench: Anomaly detection benchmark | Density-based Diffusion Methods | NeurIPS  | 2022  |[[paper]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/cf93972b116ca5268827d575f2cc226b-Abstract-Datasets_and_Benchmarks.html)
| On diffusion modeling for anomaly detection | Density-based Diffusion Methods | ICLR  | 2024  |[[paper]](https://arxiv.org/abs/2305.18593)


## 2 Complementary Generative Tasks

Anomaly synthesis and anomaly restoration have emerged as two closely related tasks that extend the scope of generative modeling in anomaly detection.
Anomaly synthesis focuses on creating realistic abnormal instances, enabling data augmentation for training, stress-testing detection models, and constructing more comprehensive evaluation benchmarks.
Anomaly restoration aims to recover anomalous inputs back to their plausible normal states, supporting interpretability, diagnostics, and even operational restoration in domains such as medical and industrial inspection.

### 2.1 Generative Anomaly Synthesis

|  Paper Title       | Method        |   Venue     | Year | Ref.  |
|---------------------|---------------------------|---------------------------------------|------------|-----|
| Cutpaste: Self-supervised learning for anomaly detection and localization | Patch-based Methods | CVPR | 2021  |[[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_CutPaste_Self-Supervised_Learning_for_Anomaly_Detection_and_Localization_CVPR_2021_paper.pdf) |
| Draem-a discriminatively trained reconstruction embedding for surface anomaly detection | Patch-based Methods | ICCV | 2021 | [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.pdf) |
| Few-shot defect segmentation leveraging abundant defect-free training samples through normal background regularization and crop-and-paste operation | Patch-based Methods | ICME |  2021 |[[paper]](https://www.computer.org/csdl/proceedings-article/icme/2021/09428468/1uim7pEfFvO) |
| Natural synthetic anomalies for self-supervised anomaly detection and localization | Patch-based Methods | ECCV | 2022  |[[paper]](https://arxiv.org/pdf/2109.15222) |
| Resynthdetect: a fundus anomaly detection network with reconstruction and synthetic features | Patch-based Methods | BMVA | 2023  |[[paper]](https://arxiv.org/pdf/2312.16470) |
| Prototypical residual networks for anomaly detection and localization | Patch-based Methods | CVPR | 2023 |[[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Prototypical_Residual_Networks_for_Anomaly_Detection_and_Localization_CVPR_2023_paper.pdf) |
| Destseg: Segmentation guided denoising student-teacher for anomaly detection | Patch-based Methods | CVPR | 2023 |[[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_DeSTSeg_Segmentation_Guided_Denoising_Student-Teacher_for_Anomaly_Detection_CVPR_2023_paper.pdf) |
| Memseg: A semi-supervised method for image surface defect detection using differences and commonalities | Patch-based Methods | Eng. Appl. Artif. Intell. | 2023 |[[paper]](https://www.sciencedirect.com/science/article/abs/pii/S0952197623000192) |
| Defect image sample generation with GAN for improving defect recognition | Adversarial Methods |  IEEE TASE | 2020 |[[paper]](https://ieeexplore.ieee.org/abstract/document/9000806) |
| Defect-GAN: High-fidelity defect synthesis for automated defect inspection | Adversarial Methods |  WACV | 2021  |[[paper]](https://openaccess.thecvf.com/content/WACV2021/papers/Zhang_Defect-GAN_High-Fidelity_Defect_Synthesis_for_Automated_Defect_Inspection_WACV_2021_paper.pdf) |
| Few-shot defect image generation via defect-aware feature manipulation | Adversarial Methods | AAAI | 2023  |[[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/25132) |
| Diversified and multi-class controllable industrial defect synthesis for data augmentation and transfer | Adversarial Methods | CVPR  | 2023  |[[paper]](https://openaccess.thecvf.com/content/CVPR2023W/VISION/papers/Wei_Diversified_and_Multi-Class_Controllable_Industrial_Defect_Synthesis_for_Data_Augmentation_CVPRW_2023_paper.pdf) |
| Findiff: Diffusion models for financial tabular data generation   | Conditional Diffusion-based Methods |  ICAIF | 2023 |[[paper]](https://dl.acm.org/doi/pdf/10.1145/3604237.3626876) |
| Realnet: A feature selection network with realistic synthetic anomaly for anomaly detection | Conditional Diffusion-based Methods |  CVPR | 2024 |[[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_RealNet_A_Feature_Selection_Network_with_Realistic_Synthetic_Anomaly_for_CVPR_2024_paper.pdf) |
| Anomalydiffusion: Few-shot anomaly image generation with diffusion model | Conditional Diffusion-based Methods |  AAAI |  2024 |[[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/28696) |
|  Video Anomaly Detection via Spatio-Temporal Pseudo-Anomaly Generation: A Unified Approach  | Conditional Diffusion-based Methods | CVPRW  | 2024  |[[paper]](https://openaccess.thecvf.com/content/CVPR2024W/VAND/papers/K._Video_Anomaly_Detection_via_Spatio-Temporal_Pseudo-Anomaly_Generation__A_Unified_CVPRW_2024_paper.pdf) |
| Netdiffus: Network traffic generation by diffusion models through time-series imaging | Conditional Diffusion-based Methods | Computer Networks  |  2024 |[[paper]](https://www.sciencedirect.com/science/article/abs/pii/S1389128624004481) |
| Few-shot defect image generation based on consistency modeling | Conditional Diffusion-based Methods |  ECCV | 2024  |[[paper]](https://arxiv.org/pdf/2408.00372?) |
| Unseen visual anomaly generation  | Conditional Diffusion-based Methods |  CVPR | 2025  |[[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_Unseen_Visual_Anomaly_Generation_CVPR_2025_paper.pdf) |


### 2.2 Generative anomaly restoration

|  Paper Title       | Method        |   Venue     | Year | Ref.  |
|---------------------|---------------------------|---------------------------------------|------------|-----|
| Attribute restoration framework for anomaly detection | Attribute-based Methods |  IEEE TMM |  2020 |[[paper]](https://ieeexplore.ieee.org/abstract/document/9311201) |
| Multi-scale cross-restoration framework for electrocardiogram anomaly detection | Attribute-based Methods | MICCAI  | 2023  |[[paper]](https://arxiv.org/pdf/2308.01639) |
| Simple and effective frequency-aware image restoration for industrial visual anomaly detection | Attribute-based Methods | AEI  | 2025  |[[paper]](https://www.sciencedirect.com/science/article/abs/pii/S1474034624007158) |
| Semantic image inpainting with deep generative models | GAN-inversion-based Methods |  CVPR |  2017 |[[paper]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Yeh_Semantic_Image_Inpainting_CVPR_2017_paper.pdf) |
| Exploiting deep generative prior for versatile image restoration and manipulation | GAN-inversion-based Methods |  IEEE TPAMI |  2021 |[[paper]](https://ieeexplore.ieee.org/abstract/document/9547753) |
| Dual-path image inpainting with auxiliary GAN inversion | GAN-inversion-based Methods | CVPR  | 2022 |[[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Dual-Path_Image_Inpainting_With_Auxiliary_GAN_Inversion_CVPR_2022_paper.pdf) |
| Uni-3dAD: Gan-inversion aided universal 3d anomaly detection on model-free products | GAN-inversion-based Methods |  Expert Systems with Applications |  2025 |[[paper]](https://www.sciencedirect.com/science/article/abs/pii/S0957417425002878) |
| RGI: robust GAN-inversion for mask-free image inpainting and unsupervised pixel-wise anomaly detection | GAN-inversion-based Methods | ICLR  | 2023  |[[paper]](https://openreview.net/forum?id=1UbNwQC89a) |
|  Anomaly detection with conditioned denoising diffusion models   | Diffusion-based Methods |  DAGM GCPR |  2024 | [[paper]](https://arxiv.org/pdf/2305.15956) |
| Disyre: Diffusion-inspired synthetic restoration for unsupervised anomaly detection| Diffusion-based Methods | ISBI  | 2024  |[[paper]](https://arxiv.org/pdf/2311.15453) |
| Diffusionad: Norm-guided one-step denoising diffusion for anomaly detection | Diffusion-based Methods | IEEE TPAMI  | 2025  |[[paper]](https://ieeexplore.ieee.org/abstract/document/11005495) |



## 3 Applications

### 3.1 Industrial manufacturing
...

### 3.2 Medical field
...

### 3.3 Finance market

...

### 3.4 Cybersecurity

...
