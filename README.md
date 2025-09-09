# Deep Convolutional Generative Adversarial Networks Study

*A Comprehensive Study on CIFAR-10 and CelebA Datasets*  
**Author:** Sia Khorsand ([@siakhorsand](https://github.com/siakhorsand))  
**University of California, San Diego**

---

## Abstract

Generative Adversarial Networks (GANs) have emerged as a powerful method in unsupervised learning, demonstrating remarkable capabilities in generating realistic synthetic data.  

This study implements and analyzes **Deep Convolutional GANs (DC-GANs)** on the **CIFAR-10** and **CelebA** datasets. I investigate:

- Activation functions (**ReLU vs. ELU**)  
- Optimization strategies  
- Hyperparameter configurations  

Key findings:

- **ReLU activations consistently outperform ELU**, producing sharper, higher-quality outputs.  
- **Learning rate asymmetry** (higher for the discriminator) is crucial for CIFAR-10 but less so for CelebA.  
- Dataset structure strongly influences optimization requirements.  

These results provide practical guidelines for implementing DC-GANs across different image domains.

---

## 1. Introduction

Since their introduction by Goodfellow et al. (2014), **GANs** have revolutionized generative modeling with their adversarial training framework:

- **Generator (G):** creates synthetic samples from noise.  
- **Discriminator (D):** distinguishes real from synthetic samples.  

> GANs can be thought of as *an arm wrestling match between two neural networks.*

### Why DC-GANs?
- Incorporation of **convolutional layers**, **batch normalization**, and **specific activations** improved stability.  
- Addressed early issues like **mode collapse**.  
- Proven effective for high-resolution image generation.  

This study investigates **dataset-specific behaviors** of DC-GANs across **CIFAR-10 (objects)** and **CelebA (faces)**.

---

## 2. Methodology

### 2.1 Architecture Design
- **Generator (G):**
  - Input: random noise → dense layer → reshaped feature map.  
  - Progressive upsampling via **transposed convolutions**.  
  - Activations: ReLU / ELU in hidden layers, Tanh in output.  
- **Discriminator (D):**
  - Input: image → convolutional downsampling.  
  - Activations: LeakyReLU throughout.  
  - Includes **batch normalization** for stability.  

### 2.2 Training Strategy
- **Objective:** Minimax loss (standard GAN formulation).  
- **Stabilization techniques:**
  - Spectral normalization (CIFAR-10)  
  - Exponential Moving Averages (EMA)  
  - Instance noise injection with decay  
  - Label smoothing  
  - Mixed precision training (CIFAR-10)  
  - Careful weight initialization & learning rate tuning  

### 2.3 Experimental Design
- Variables explored:
  - **Activation functions:** ReLU vs. ELU  
  - **Learning rates:** symmetric vs. asymmetric (higher for D)  
  - **Architectures & seeds**  
- Full logging of loss dynamics and generated outputs.

---

## 3. Datasets

### CIFAR-10
- **60,000 images (32×32)** across 10 object categories.  
- Balanced: 5k train / 1k test per class.  
- Challenges: high diversity, low resolution.  
- Preprocessing: normalization → `[-1, 1]`, augmentation (flips, rotations).  

### CelebA
- **200k+ celebrity faces** (40 attributes each).  
- Subset of ~50k images (64×64, cropped & aligned).  
- Challenges: requires **fine detail** (skin, symmetry, textures).  
- Preprocessing: center crop, normalization, quality filtering.

---

## 4. CIFAR-10 Results & Discussion

### 4.1 Training Dynamics
- **ELU models:**  
  - Rapid imbalance (D dominates).  
  - Blurry, desaturated samples (mode collapse).  
  - IS = **2.87 ± 0.98**.  

- **ReLU w/ asymmetric LR (G=1e-4, D=2e-4):**  
  - Balanced adversarial training.  
  - Sharp, diverse, realistic samples.  
  - IS = **5.49 ± 1.8**.  

- **ReLU w/ symmetric LR (both 1e-4):**  
  - Generator stagnation, poor diversity.  
  - Many gray/blurry outputs.  

### 4.2 Architectural Insights
- **ReLU:** Maintains aggressive signal propagation → better detail.  
- **ELU:** Saturation leads to over-smoothed textures.  

### 4.3 Hyperparameter Optimization
- Asymmetric LR critical for CIFAR-10.  
- Symmetric LR = stagnation.  

### 4.4 Sample Quality
- **Visuals:** ReLU samples had sharp edges, textures, object recognizability.  
- **Metrics:** Clear IS advantage for ReLU + asymmetric LR.  

---

## 5. CelebA Results & Discussion

### 5.1 Architectural Impact
- **ReLU:**  
  - High-fidelity faces, sharp edges, realistic textures.  
  - Preserves skin, hair, and fine-grained details.  
- **ELU:**  
  - Softer textures, less detail.  
  - Still learns basic facial structure but lacks realism.  

### 5.2 Hyperparameter Optimization
- ReLU sensitive to LR; best: **D=0.0003, G=0.0002**.  
- ELU optimal configs still underperform compared to ReLU.  

### 5.3 Sample Quality
- **ReLU:** IS = **6.82 ± 1.4** → sharp, diverse, photorealistic faces.  
- **ELU:** IS = **4.91 ± 1.2** → softer, less detailed faces.  

---

## 6. Analysis & Conclusion

### 6.1 Comparative Analysis
- **CIFAR-10:** performance depends heavily on LR asymmetry.  
- **CelebA:** more robust to LR, but still ReLU > ELU.  
- Dataset constraints matter:
  - Faces (CelebA) → more structurally guided.  
  - Objects (CIFAR-10) → require fine-tuned adversarial balance.  

### 6.2 Training Lessons
- **ReLU > ELU** across both datasets.  
- **Learning rate asymmetry** essential for diverse datasets (CIFAR-10).  
- Early warning signs:
  - Rapid discriminator dominance.  
  - Uniform/blurry generated samples.  

### 6.3 Practical Guidelines
- Use **ReLU activations**.  
- For object datasets (CIFAR-10-like):  
  - Asymmetric LR (D:G ≈ 2:1).  
  - Monitor early training losses.  
- For face datasets (CelebA-like):  
  - Balanced LR between 1e-4 and 5e-4 is sufficient.  

**Optimal Results:**  
- CIFAR-10 (ReLU, asymmetric LR): **IS = 5.49 ± 1.8**  
- CelebA (ReLU, tuned LR): **IS = 6.82 ± 1.4**  

---

## Conclusion & Future Work

This study demonstrates that:

- **ReLU is superior to ELU** in DCGANs.  
- **Dataset structure strongly influences optimization needs.**  
- Proper **learning rate strategy** is as important as activation choice.  

### Future Directions
- Explore **Progressive GANs** and **StyleGAN** with similar experiments.  
- Investigate **transfer learning** between datasets.  
- Extend findings to **larger-scale, higher-resolution image domains**.

---

## References

- Goodfellow, I., et al. (2014). *Generative adversarial nets*. NIPS.  
- Radford, A., Metz, L., & Chintala, S. (2015). *Unsupervised representation learning with DCGANs*. arXiv:1511.06434.  
- Krizhevsky, A., & Hinton, G. (2009). *Learning multiple layers of features from tiny images*.  
- Liu, Z., et al. (2015). *Deep learning face attributes in the wild*. ICCV.  

---
