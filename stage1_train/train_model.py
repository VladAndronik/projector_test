import os.path
import re

import numpy as np
import pandas as pd
from joblib import dump
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

PATH_DATA = 'data/'

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z +_-]')
STOPWORDS = stopwords.words('english')
RANDOM_STATE = 17

def text_prepare(text):
    """
        text: a string

        return: modified initial string
    """
    if pd.isna(text):
        return ''

    text = text.strip().lower()
    text = re.sub(REPLACE_BY_SPACE_RE, ' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(BAD_SYMBOLS_RE, '', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = re.sub('-', ' ', text)  # if there is defise split the words
    text = ' '.join([s for s in text.split() if s not in STOPWORDS])  # delete stopwords from text
    text = [token for token in text.split() if not token.isdigit()]  # remove digits
    text = ' '.join([s for s in text])

    return text


def main():
    data = pd.read_csv(os.path.join(PATH_DATA, 'train.csv'))
    data['text_prepared'] = data['excerpt'].apply(text_prepare)

    # split into different categories by how hard to identify
    data['complexity'] = 0
    data.loc[(data['standard_error'] >= 0.45) & (data['standard_error'] < 0.55), 'complexity'] = 1
    data.loc[(data['standard_error'] >= 0.55), 'complexity'] = 2

    # train/val split
    train, val = train_test_split(data, test_size=.1, stratify=data['complexity'], random_state=RANDOM_STATE)

    X_train = [t for t in train['text_prepared']]
    X_val = [t for t in val['text_prepared']]

    y_train = train['target'].values
    y_val = val['target'].values

    print('Vectorizing words...')
    tfidf_vectorizer = TfidfVectorizer(min_df=3, max_df=.9, ngram_range=(1, 2), token_pattern='\S+')
    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_val = tfidf_vectorizer.transform(X_val)

    # predict
    print("Fitting the model...")
    model = LassoCV(cv=3, random_state=RANDOM_STATE, n_jobs=3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    print(f"RMSE = {np.sqrt(mean_squared_error(y_pred, y_val))}")

    dump(model, 'model.joblib')


if __name__ == '__main__':
    main()

# scores:
# linear_regression: 0.8904
# lassocv: 0.8356
# randomforest: 0.8692
