# 🧑‍🤝‍🧑 Gender Classification Using Hybrid Deep Learning & Machine Learning

---

## 📌 Overview

This project develops an efficient **gender classification system** from facial images using a hybrid approach that combines **Convolutional Neural Networks (CNN)** for deep feature extraction with classical **Machine Learning classifiers** (SVM, Random Forest, XGBoost) for the final prediction.

The system is trained and evaluated on the **UTKFace dataset** — a large-scale, real-world facial image dataset with diverse age, race, and gender demographics.

---


## 🏗️ System Architecture

```
Input Facial Image (Grayscale → 128×128px)
        │
        ▼
  MobileNetV2 (Transfer Learning, ImageNet)
  ┌─── Grayscale → 3-channel Conv Layer
  ├─── Frozen Base Layers (Feature Extraction)
  ├─── Global Average Pooling
  ├─── Dense (ReLU) → Dropout
  └─── Output: Sigmoid (Male / Female)
        │
        ├─── CNN-only Classification
        │
        └─── Deep Feature Extraction (GAP Layer)
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
         SVM    Random    XGBoost
               Forest
          └─────────┴─────────┘
                    │
             Ensemble Voting
```

---

## 🧠 Models Used

### Deep Learning
| Model | Architecture | Purpose |
|---|---|---|
| CNN | MobileNetV2 (fine-tuned) | End-to-end binary gender classification |

### Hybrid Approach (CNN Features + ML)
| Model | Key Strength |
|---|---|
| SVM | Effective in high-dimensional spaces; robust against overfitting |
| Random Forest | Handles noisy data; reduces overfitting via ensemble voting |
| XGBoost | Gradient boosting; handles missing data; high predictive power |
| Ensemble (Stacking) | Combines base model predictions via a meta-learner |

### Classical ML Baseline
- **LBP (Local Binary Pattern)** features + SVM / Random Forest / XGBoost for comparison

---

## 📊 Results

### CNN Model Training Metrics

| Epoch | Accuracy | Loss | Val Accuracy | Val Loss |
|---|---|---|---|---|
| 1 | 0.7643 | 0.4876 | 0.5240 | 1.3169 |
| 4 | 0.9514 | 0.1228 | 0.7155 | 0.5672 |
| 7 | 0.9765 | 0.0651 | 0.7769 | 0.6543 |
| 10 | **0.9872** | **0.0402** | **0.8705** | 0.5219 |

> Training accuracy reached **98–99%**; validation accuracy stabilized at **~88–89%**. A visible gap indicates slight overfitting.

---

### Machine Learning Models (on CNN Deep Features)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.90 | 0.91 | 0.91 | 0.91 | 0.97 |
| SVM | 0.90 | 0.91 | 0.90 | 0.90 | 0.97 |
| **XGBoost** | **0.91** | **0.92** | **0.91** | **0.91** | **0.97** |

> **XGBoost** was the best-performing classifier with the highest accuracy, precision, and F1-score.  
> All models achieved an excellent **ROC-AUC of 0.97**, demonstrating strong class separability.

---

### Real-World Validation (30 test images)

| Gender | Correct | Total | Accuracy |
|---|---|---|---|
| Male | 13 | 15 | 86.7% |
| Female | 11 | 15 | 73.3% |

---

## 🗂️ Dataset

**UTKFace Dataset** — Large-scale facial image dataset for demographic analysis.

| Property | Details |
|---|---|
| Total Images | 23,000+ |
| Labels | Age, Gender, Race (embedded in filename) |
| Gender Labels | 0 = Female, 1 = Male |
| Age Range | 0–116 years |
| Race Groups | White, Black, Asian, Indian, Others |
| Conditions | Unconstrained (real-world poses, lighting, occlusions) |
| Used Labels | Gender only (binary classification) |

Filename format: `[Age]_[Gender]_[Race]_[DateTime].jpg`

---

## ⚙️ Method & Approach

1. **Preprocessing** — Images loaded in grayscale, resized to `128×128`, pixel values normalized to `[0, 1]`
2. **CNN Training** — MobileNetV2 fine-tuned with grayscale-to-3-channel conversion layer; sigmoid output for binary classification
3. **Feature Extraction** — Deep features extracted from the **Global Average Pooling (GAP)** layer
4. **Hybrid Classification** — GAP features passed to SVM, Random Forest, XGBoost
5. **Classical Baseline** — LBP handcrafted features used with the same ML classifiers for comparison
6. **Evaluation** — Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC

---

## 🛠️ Tech Stack

- **Deep Learning:** TensorFlow / Keras, MobileNetV2
- **Machine Learning:** Scikit-learn (SVM, Random Forest), XGBoost
- **Image Processing:** OpenCV, NumPy
- **Feature Extraction:** LBP (scikit-image), CNN GAP layer
- **Visualization:** Matplotlib, Seaborn

---

🌐 Deployment
The system is deployed as an interactive web application on Hugging Face Spaces using Gradio, supporting:

🔗 Live Demo: https://huggingface.co/spaces/Nikhil0702/Gender_Classification_Using_Hybrid_Approach/tree/main


