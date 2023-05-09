import os
import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

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
random.seed(42)
random.shuffle(data)

texts, labels = zip(*data)

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# extract n-grams from the text data
ngram_vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=1)
X_train_ngrams = ngram_vectorizer.fit_transform(X_train)
X_test_ngrams = ngram_vectorizer.transform(X_test)

# train a classifier
clf = MultinomialNB()
clf.fit(X_train_ngrams, y_train)

# make predictions on the test set
y_pred = clf.predict(X_test_ngrams)

# evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))