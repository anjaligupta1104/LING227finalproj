import os
import random
from collections import defaultdict
from nltk import ngrams
from nltk.lm import MLE, Lidstone, KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

def preprocess(text):
    tokens = word_tokenize(text)

    # remove stopwords
    # tokens = [token for token in tokens if token not in stopwords.words("english")]
    # removing stopwords didn't improve accuracy

    # stem
    stemmer = SnowballStemmer("english")
    tokens = [stemmer.stem(token) for token in tokens]

    # join tokens back together
    text = " ".join(tokens)

    return text


def load_data(data_dir, label):
    texts = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".txt"):
            with open(os.path.join(data_dir, file_name), "r", encoding="utf-8") as file:
                text = file.read()
                text = preprocess(text)
                texts.append((text, label))
    return texts

impaired_texts = load_data("./data/impaired", "impaired")
not_impaired_texts = load_data("./data/not_impaired", "not_impaired")
data = impaired_texts + not_impaired_texts
random.seed(25)
random.shuffle(data)

texts, labels = zip(*data)

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.1, random_state=42)

# train an n-gram language model for each class
n = 2
class_text = defaultdict(str)

for text, label in zip(X_train, y_train):
    class_text[label] += " " + text

class_models = {}
for label, text in class_text.items():
    train_data, padded_sents = padded_everygram_pipeline(n, [text])
    model = MLE(n)
    model.fit(train_data, padded_sents)

    # tried Lidstone and Kneser-Ney, but was unable to fix bugs with both
    # kneser_ney_model = KneserNeyInterpolated(n)
    # kneser_ney_model.fit(train_data, padded_sents)

    class_models[label] = model

# classify the test texts
y_pred = []
for text in X_test:
    likelihoods = {}
    for label, model in class_models.items():
        score = 0
        for ng in ngrams(text, n):
            score += model.logscore(ng[-1], ng[:-1])
        likelihoods[label] = score
    y_pred.append(max(likelihoods, key=likelihoods.get))

# evaluate the classifier
accuracy = sum(pred == true for pred, true in zip(y_pred, y_test)) / len(y_test)
print(f"Accuracy: {accuracy:.2f}")