# Enron Email Spam Detection: End-to-End NLP Benchmark (Baseline GAP, 1D CNN, GRU/RNN & Transformer)

An end-to-end, scalable TensorFlow/Keras natural language processing (NLP) pipeline that builds, trains, evaluates, and compares four distinct deep learning architectures on the **Enron Email Spam Dataset**—complete with threshold tuning, confusion matrix error analysis, and a serializable end-to-end inference pipeline.

## Final 4-Model Test Set Benchmark (1,160 Emails)

| Rank | Model Architecture | Test Loss | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Parameters |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Model 1: Baseline (GAP)** | **0.0459** | **98.97%** | **98.81%** | **98.81%** | **98.81%** | **1,288,321** |
| **2** | **Model 4: Transformer** | 0.0466 | 98.53% | 98.02% | 98.61% | 98.32% | 1,387,905 |
| **3** | **Model 2: 1D CNN** | 0.0512 | 98.28% | 97.82% | 98.21% | 98.02% | 1,325,249 |
| **4** | **Model 3: GRU (RNN)** | 0.1449 | 96.03% | 94.72% | 96.22% | 95.46% | 1,321,473 |

## Error Analysis & Decision Threshold Tuning (Model 1)
* **Confusion Matrix (Threshold = 0.50):** 651 True Negatives, 497 True Positives, **6 False Positives**, 6 False Negatives.
* **Threshold Tuning (`0.85`):** Reduces False Positives (legitimate emails wrongly blocked) from **6 down to 2** (99.69% specificity).
* **Domain Drift Finding:** Live inference testing on modern credential-phishing templates demonstrated why models trained on historical 2000–2002 corporate emails require periodic retraining on modern phishing examples.
