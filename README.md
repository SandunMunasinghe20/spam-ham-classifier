# Spam Ham Classifier

A machine learning project to detect spam emails using the Spam Ham dataset. This project uses **TF-IDF vectorization** and **Linear Support Vector Classifier (LinearSVC)**. It also includes optional PyTorch support for experimenting with deep learning.

## Dataset

- Source: Spam Ham Dataset (`spam_ham_dataset.csv`)
- Shape: (5171, 4)
- Features: 
  - `label` : 'ham' or 'spam'
  - `text` : email content
  - `label_num` : 0 (ham) / 1 (spam)

After balancing, the dataset contains 2998 emails (1499 ham + 1499 spam).

## Model

- Vectorization: `TfidfVectorizer` (max_features=10000, stop_words='english')  
- Classifier: `LinearSVC` from scikit-learn  
- Train/Test split: 80/20  
- Accuracy achieved: **98%**  

## Requirements

```txt
pandas
numpy
scikit-learn
matplotlib
seaborn
torch
