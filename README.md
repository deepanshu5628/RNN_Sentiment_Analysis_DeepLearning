# RNN Sentiment Analysis - Deep Learning

Sentiment analysis on the IMDB movie reviews dataset using a Recurrent Neural Network (RNN) built with PyTorch.

## Dataset

- **Source:** IMDB Dataset (50,000 movie reviews)
- **Task:** Binary classification (positive / negative sentiment)

## Pipeline

1. **Load & Validate Data** – Remove duplicates and null values
2. **Text Preprocessing:**
   - Lowercasing
   - URL removal
   - HTML tag removal
   - Punctuation removal
   - Stopword removal (NLTK)
   - Stemming (Porter Stemmer)
3. **Feature Extraction** – TF-IDF vectorization (max 2000 features)
4. **Label Encoding** – Encode sentiment labels
5. **Train/Test Split** – 80/20 split
6. **Model** – Single-layer RNN (128 hidden units) with a fully connected output layer
7. **Training** – Adam optimizer, Binary Cross-Entropy loss, 100 epochs
8. **Evaluation** – Accuracy on test set

## Tech Stack

- Python 3.13
- PyTorch
- NLTK
- Scikit-learn
- Pandas / NumPy

## Usage

```bash
pip install torch nltk scikit-learn pandas numpy
```

Run the Jupyter notebook to train and evaluate the model.
