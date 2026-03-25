# Diabetes Detection from Clinical Notes using NLP

## Overview
This project develops a machine learning pipeline to automatically detect diabetes-related cases from clinical notes using Natural Language Processing (NLP). The system processes unstructured medical text and classifies whether a given note indicates diabetes.

---

## Problem Statement
Healthcare systems generate large volumes of unstructured clinical text, making manual analysis time-consuming and error-prone. This project addresses the problem of identifying diabetes-related cases from clinical notes to support faster and more consistent clinical decision-making.

---

## Dataset
- Source: Hugging Face – `birgermoell/icd10-clinical-notes`
- Train set: 1441 records  
- Test set: 361 records  
- Target: Binary classification (diabetes vs non-diabetes)
- Labeling strategy: ICD-10 codes starting with `E10` and `E11` were mapped to diabetes

Note: The dataset is synthetic, which simplifies the task but limits real-world generalizability.

---

## System Design
The system follows a simple NLP pipeline:

1. **Data Preprocessing**
   - Extraction of relevant fields
   - Creation of binary diabetes label

2. **Feature Engineering**
   - TF-IDF vectorization of clinical notes

3. **Modeling**
   - Logistic Regression
   - Multinomial Naive Bayes

4. **Evaluation**
   - Classification report (precision, recall, F1-score)
   - Confusion matrix

---

## Results

### Logistic Regression
- Recall (diabetes): 0.93  
- Precision (diabetes): 1.00  
- Missed 1 diabetes case  

### Naive Bayes
- Recall (diabetes): 1.00  
- Precision (diabetes): 0.93  
- 1 false positive  

---

## Model Comparison
Logistic Regression is more conservative and avoids false positives, whereas Naive Bayes prioritizes recall and detects all diabetes cases.

In a medical context, recall is more critical than precision, as failing to detect a disease can have severe consequences. Therefore, Naive Bayes is considered the more suitable model for this task.

---

## Limitations
- The dataset is synthetic and may not reflect real-world clinical complexity  
- Strong lexical cues likely simplify classification  
- Class imbalance (low number of diabetes cases)  
- Lack of ambiguous or borderline cases  

---

## Reproducibility
The full pipeline can be executed using:

```bash
python src/preprocess.py
python src/train.py
python src/evaluate.py

## Project Structure

data/
notebooks/
src/
results/
README.md
