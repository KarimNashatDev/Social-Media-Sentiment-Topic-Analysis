# Social Media Sentiment & Topic Analysis

An end-to-end NLP pipeline for analyzing sentiment and topics in Twitter data, comparing traditional ML models, deep learning architectures, and transformer-based approaches.

## Overview

This project classifies tweets into **Positive**, **Negative**, **Neutral**, and **Irrelevant** sentiment categories while also extracting latent topics using LDA. It benchmarks six different models and includes an interactive topic visualization.

## Project Structure

```
Social Media Sentiment & Topic Analysis/
├── twitter_project final.ipynb      # Main notebook (full pipeline)
├── twitter_training.csv             # Training set (~74,682 tweets)
├── twitter_validation.csv           # Validation set (1,000 tweets)
├── lda_topic_visualization.html     # Interactive pyLDAvis topic visualization
└── Video/                           # Demo/presentation video
```

## Dataset

| Column      | Description                                |
|-------------|--------------------------------------------|
| `Id`        | Unique tweet identifier                    |
| `Entity`    | Brand/game/topic the tweet references      |
| `Sentiment` | `Positive`, `Negative`, `Neutral`, `Irrelevant` |
| `Content`   | Raw tweet text                             |

Entities include: Borderlands, Facebook, Amazon, Microsoft, CS-GO, Google, FIFA, MaddenNFL, Tom Clancy's Rainbow Six, Assassin's Creed, Call of Duty, and others.

## Pipeline

1. **Data Loading & Cleaning** — regex-based text normalization, lowercasing, stopword removal, tokenization
2. **Exploratory Data Analysis** — sentiment distribution plots, word clouds, text length analysis
3. **Topic Modeling** — Latent Dirichlet Allocation (18 topics) with interactive pyLDAvis visualization
4. **Feature Engineering** — tokenized sequences with padding, plus engineered features (text length, hashtag/mention counts)
5. **Model Training & Evaluation** — six models compared on accuracy, precision, recall, and F1-score

## Models & Results

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.78     | —         | —      | —        |
| Linear SVC          | 0.78     | —         | —      | —        |
| LSTM                | 0.27     | 0.07      | 0.25   | 0.11     |
| CNN                 | 0.95     | 0.95      | 0.95   | 0.95     |
| **Attention-LSTM**  | **0.96** | **0.96**  | **0.97** | **0.96** |
| DistilBERT          | 0.93     | 0.94      | 0.93   | 0.93     |

The **Attention-LSTM** achieved the best overall performance, surpassing even the transformer-based DistilBERT.

### Key Technical Details

- **Traditional ML**: Logistic Regression and Linear SVC with `GridSearchCV` hyperparameter tuning
- **CNN**: Conv1D + GlobalMaxPooling + Dense architecture
- **Attention-LSTM**: LSTM(128) with a custom `AttentionLayer` for weighted sequence representation
- **DistilBERT**: Fine-tuned `distilbert-base-uncased` via HuggingFace Transformers (lr=2e-5, 4 epochs)
- **Class imbalance** handled via computed class weights
- **Early stopping** applied to deep learning models

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
wordcloud
nltk
tensorflow
transformers
pyLDAvis
keras
```

## How to Run

1. Clone this repository
2. Open `twitter_project final.ipynb` in Jupyter Notebook or JupyterLab
3. Run all cells sequentially

> **Note**: Deep learning models (CNN, Attention-LSTM, DistilBERT) require a GPU for reasonable training times.

## Topic Visualization

Open `lda_topic_visualization.html` in a browser to interactively explore the 18 discovered topics. The visualization uses pyLDAvis to show topic distributions, top terms, and inter-topic distances.
