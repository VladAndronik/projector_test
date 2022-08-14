from joblib import load

from stage1_train.train_model import text_prepare

tfidf_vect = load('trained_models/tfidf.joblib')
model = load('trained_models/model.joblib')


def make_prediction(text: str):
    x = [text_prepare(text)]
    x = tfidf_vect.transform(x)
    pred = model.predict(x)

    return {'results': pred[0]}


if __name__ == '__main__':
    pred = make_prediction("comprehensive relevance to the essential part of the plot")
    print(pred)
