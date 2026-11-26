### [Vehicle Image Identification: End-to-End Computer Vision Benchmark (Baseline MLP, 2D CNN, Transfer Learning & Hybrid ViT)](./vehicle-identification-benchmark/)
An end-to-end, scalable TensorFlow/Keras computer vision pipeline that builds, trains, evaluates, and compares six deep learning architectures across **48 imbalanced vehicle models**—complete with `tf.data` parallel streaming, model-internal preprocessing, confusion matrix error analysis, CPU-to-GPU mixed-precision scaling (`USE_GPU`), and a serializable `.keras` inference pipeline.

#### Final Multi-Model Test Set Benchmark (752 Vehicle Images)

| Rank | Model Architecture | Test Accuracy (%) | Macro F1-Score (%) | Weighted F1-Score (%) | Key Architectural Takeaway |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **1** | **MobileNetV2 (Fine-Tuned Top 30)** | **79.12%** | **68.93%** | **77.64%** | **Best Overall** — Unfreezing top 30 layers (with frozen BatchNorm) captured fine grille/headlight cues. |
| **2** | **MobileNetV2 (Frozen Feature Extractor)** | **72.87%** | **64.12%** | **70.94%** | Strong transfer-learning baseline; struggles to separate look-alike model years. |
| **3** | **Transfer CNN (Augmentation + Class Weights)** | **65.29%** | **57.95%** | **65.75%** | Heavy regularization & class weighting; requires full GPU epoch schedules (`20+` epochs) to converge. |
| **4** | **Model 1: Baseline MLP (Pooled Dense)** | **15.82%** | **3.46%** | **8.46%** | Non-spatial baseline proving convolutional inductive bias is essential for vision tasks. |
| **5** | **Model 2: Custom 2D CNN (From Scratch)** | **8.91%** | **0.34%** | **1.46%** | Requires large-scale data (`100k+` images) and `30+` GPU epochs to overcome random initialization. |
| **6** | **Model 4: Hybrid CNN-ViT (From Scratch)** | **8.91%** | **0.34%** | **1.46%** | Patch-token + multi-head self-attention sequence model engineered for large-scale GPU training. |

#### Error Analysis & Scalability Findings (Fine-Tuned MobileNetV2)
- **Fine-Grained & Badge-Engineered Confusions:** Top misclassifications occurred between visually identical Kei cars and sibling models (`Suzuki_alto_2007` ↔ `Daiatsu_Core`, `Suzuki_Every` ↔ `Daiatsu_Hijet`, `Toyota_Passo` ↔ `Daiatsu_Mira`), reducing maximum pair errors from 6 down to 2–4 after fine-tuning.
- **Class Imbalance Impact:** Across a 21:1 class imbalance ratio (269 images in the largest class vs. 13–16 in the rarest), fine-tuning lifted **Macro F1 by +4.81%** (`64.12%` → `68.93%`), proving deeper layer adaptation improves rare-vehicle recognition without sacrificing overall accuracy.
- **CPU-to-GPU Scalability Design:** Built with lazy path-based `tf.data` loading, capped shuffle buffers (`min(len(X), 10000)`), model-embedded preprocessing layers, and a single `USE_GPU` flag to toggle `mixed_float16` precision when scaling to massive datasets.