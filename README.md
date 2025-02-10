# Gender Classification by Spectral Data

## Overview
This repository contains the implementation of gender classification using **spectral voice features**. Various models such as **Logistic Regression, SVMs (Linear & RBF), and LSTM** are evaluated based on **accuracy, recall, precision, and F1-score** with cross-validation.

## Models Evaluated
| Model              | Test Accuracy | Cross-Validation Accuracy (Mean) | Precision (F/M)         | Recall (F/M)         | F1-Score (Macro Avg) | Key Insights |
|--------------------|--------------|----------------------------------|-------------------------|----------------------|----------------------|--------------|
| Logistic Regression | 98.91%       | 98.91%                           | High                    | 98.45% / 98.78%      | 98.61%               | No overfitting, robust generalization |
| Decision Tree     | 93.28%       | 93.13%                           | 91.35% / 94.33%         | 89.69% / 95.28%      | 92.66%               | Better at classifying males, potential recall improvement for females |
| Linear SVM        | 99.35%       | 99.22%                           | Very High               | 99.10% / 99.45%      | 99.27%               | Strong performance, but slightly outperformed by RBF SVM |
| RBF SVM          | 99.94%       | 99.94%                           | Extremely High          | 99.89% / 99.98%      | 99.94%               | Best model overall, captures non-linear patterns |

## Decision Matrices
| Model               | TN   | FP  | FN  | TP   |
|---------------------|------|-----|-----|------|
| Logistic Regression | 1100 | 54  | 34  | 2042 |
| Decision Tree      | 1035 | 119 | 213 | 1863 |
| Linear SVM         | 1130 | 24  | 15  | 2061 |
| RBF SVM           | 1149 | 5   | 2   | 2074 |

## Feature Importance
- MFCCs (Mel Frequency Cepstral Coefficients) are the most relevant features for gender classification.

## How to Use
```bash
git clone https://github.com/your-repo/gender-classification-spectral.git
cd gender-classification-spectral
pip install -r requirements.txt
python lstm_crossval.py
```

## Key Takeaways
✅ **Best Model:** RBF SVM (99.94% accuracy)  
✅ **Most Interpretable Model:** Logistic Regression (easier to understand feature contributions)  
⚠️ **Potential Bias:** Decision Tree favors male classification slightly more than female  
📌 **Next Steps:** Consider ensemble learning, fairness adjustments, and further hyperparameter tuning.

## Context
Developed as part of an **Intro to Machine Learning** course.
