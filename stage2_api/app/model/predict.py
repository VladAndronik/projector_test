from joblib import load
from pathlib import Path
import re
from nltk.corpus import stopwords


BASE_DIR = Path(__file__).resolve(strict=True).parent

tfidf_vect = load(f'{BASE_DIR}/trained_models/tfidf.joblib')
model = load(f'{BASE_DIR}/trained_models/model.joblib')

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z +_-]')
STOPWORDS = stopwords.words('english')
RANDOM_STATE = 17

def text_prepare(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.strip().lower()
    text = re.sub(REPLACE_BY_SPACE_RE, ' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(BAD_SYMBOLS_RE, '', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = re.sub('-', ' ', text)  # if there is defise split the words
    text = ' '.join([s for s in text.split() if s not in STOPWORDS])  # delete stopwords from text
    text = [token for token in text.split() if not token.isdigit()]  # remove digits
    text = ' '.join([s for s in text])

    return text


def make_prediction(text: str):
    x = [text_prepare(text)]
    x = tfidf_vect.transform(x)
    pred = model.predict(x)

    return {'results': pred[0]}


if __name__ == '__main__':
    pred = make_prediction("comprehensive relevance to the essential part of the plot")
    print(pred)
