# SST-2 Sentiment Model Audit

## What this is
An end-to-end audit of a pre-trained sentiment analysis model 
on the SST-2 benchmark dataset.

## What I found
- Training data has a 55.8/44.2 positive/negative imbalance
- Model achieves 91.1% overall accuracy but is asymmetric:
  93% true positive rate vs 89% true negative rate
- Model is overconfident on errors — failures cluster at 0.99 confidence,
  meaning confidence scores cannot be used as a reliability signal
- Performance degrades on complex sentences containing negation,
  sarcasm, and concessive structure

## Why it matters
A model with these properties would systematically under-catch 
negative customer feedback and cannot be safely filtered by confidence threshold.

## Stack
Python · HuggingFace datasets · transformers · PyTorch (MPS) · pandas · seaborn

## How to run
pip install -r requirements.txt
jupyter notebook audit.ipynb
