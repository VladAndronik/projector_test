## Training model
1. Preprocess text first, removing bad symbols and stop words
2. Using TF-IDF for vectorizing words with 2 ngrams
3. Split data into train and val sets with stratification on complexity derived from standard error.
4. Train Lasso regression.

## Reproduce
For training, this will save the trained model and vectorizer:
```python train_model.py```