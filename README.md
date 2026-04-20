# LW4_Improving-CNN-Performance

# 🌿 Medical Plants Image Classification - Model Evaluation Report

**Model:** CSC_120_Image_Classifier_Improved.keras  
**Architecture:** MobileNetV2 (Transfer Learning)  
**Dataset:** 5,001 images across 20 classes  
**Image Size:** 224×224 | **Batch Size:** 32  
**Validation Split:** 20% (1,000 images)

---

## 📊 Overall Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 86% |
| **Macro Avg Precision** | 0.87 |
| **Macro Avg Recall** | 0.87 |
| **Macro Avg F1-Score** | 0.87 |
| **Weighted Avg F1-Score** | 0.87 |

---

## A. Model Evaluation Analysis

### 1. Weakest-Performing Classes

Based on the confusion matrix and classification report:

| Class | Precision | Recall | F1-Score | Issue |
|-------|-----------|--------|----------|-------|
| `_Blumea_balsamifera` | 0.73 | 0.71 | **0.72** | Lowest overall performance |
| `Peperomia_pellucida` | 0.73 | 0.78 | **0.75** | Low precision |
| `Chili_plants` | 0.91 | 0.69 | **0.78** | Poor recall (high false negatives) |
| `Mentha_spicata` | 0.75 | 0.85 | **0.80** | Lower precision |
| `Phyllanthus_niruri_Linn` | 0.84 | 0.79 | **0.81** | Balanced but moderate |

**Key Insight:** `_Blumea_balsamifera` is the most problematic class, likely due to visual similarity with other broad-leaf plants.

### 2. Precision, Recall & F1-Score Variation

**High Performers (F1 &gt; 0.90):**
- `Akapulko` (1.00) — Perfect classification
- `Giant Taro` (1.00) — Perfect classification  
- `globe_amaranth` (0.95)
- `giant_crape_myrtle` (0.90)
- `Sesbania_grandiflora_` (0.90)
- `Lantana_camara` (0.91)
- `muskmelon` (0.91)
- `Wrightia_antidysenterica` (0.91)

**Moderate Performers (F1: 0.80-0.89):**
- Most classes fall in this range

**Low Performers (F1 &lt; 0.80):**
- `_Blumea_balsamifera`, `Peperomia_pellucida`, `Chili_plants`

**Pattern:** Classes with distinct visual features (unique leaf shapes, colors) achieve perfect scores, while visually similar species get confused.

### 3. What Low Recall Indicates

**Low recall = High false negatives** (model misses actual positives)

| Class | Recall | Interpretation |
|-------|--------|----------------|
| `Chili_plants` | 0.69 | Misses 31% of actual chili plants |
| `_Blumea_balsamifera` | 0.71 | Misses 29% of actual instances |

**Causes:**
- Class imbalance in training data
- Visual similarity with other species
- Insufficient distinctive feature learning

### 4. AUC vs. Accuracy

| Aspect | Accuracy | AUC (ROC) |
|--------|----------|-----------|
| **Measures** | Overall correct predictions | Ability to rank positive vs. negative classes |
| **Class Imbalance** | Sensitive | Robust |
| **Interpretation** | "How often right?" | "How well does it separate classes?" |

**For this model:** High AUC scores (&gt;0.85 for most classes) confirm strong discriminative ability beyond raw accuracy.

---

## B. Model Improvement

### 5. Data Augmentation Effects

**Applied Augmentations:**
```python
Sequential([
    RandomFlip,
    RandomRotation,
    RandomZoom
])
