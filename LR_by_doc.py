from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import re
import random
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from scipy.sparse import hstack
import warnings
from sklearn.exceptions import ConvergenceWarning
import numpy as np

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def preprocess(text):
    tokens = word_tokenize(text)

    # remove stopwords
    tokens = [token for token in tokens if token not in stopwords.words("english")]
    # removing stopwords didn't have effect on accuracy

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

# no additional features, just tf-idf token counts
v = TfidfVectorizer()
X_train_counts = v.fit_transform(X_train)
X_test_counts = v.transform(X_test)

model = LogisticRegression(solver = 'saga', random_state=227)
model.fit(X_train_counts, y_train)
pred_y = model.predict(X_test_counts)

print("Word counts: ", round(accuracy_score(y_test, pred_y), 4))

# additional feature helper functions
def restarted_phrases(texts):
    karets_exp = re.compile(r'<[^>]*>')
    return [[len(re.findall(karets_exp, text))] for text in texts]

def hesitant(texts):
    return [[text.count('&-')] for text in texts]

def grammatical_error(texts):
    return [[text.count('[*]')] for text in texts]

def reduced_words(texts):
    return [[text.count('&~')] for text in texts]

def um(texts):
    return [[text.count('um')] for text in texts]

# extracting features from data
X_train_restart = restarted_phrases(X_train)
X_test_restart = restarted_phrases(X_test)

X_train_hesitant = hesitant(X_train)
X_test_hesitant = hesitant(X_test)

X_train_gram = grammatical_error(X_train)
X_test_gram = grammatical_error(X_test)

X_train_reduced = reduced_words(X_train)
X_test_reduced = reduced_words(X_test)

X_train_um = um(X_train)
X_test_um = um(X_test)

# adding each feature in turn
X_train_1 = hstack([X_train_counts, X_train_restart])
X_test_1 = hstack([X_test_counts, X_test_restart])

model = LogisticRegression(solver = 'saga', random_state=227)
model.fit(X_train_1, y_train)
pred_y = model.predict(X_test_1)

print("Restarted phrases: ", round(accuracy_score(y_test, pred_y), 4))


X_train_2 = hstack([X_train_counts, X_train_hesitant])
X_test_2 = hstack([X_test_counts, X_test_hesitant])

model2 = LogisticRegression(solver = 'saga', random_state=227)
model2.fit(X_train_2, y_train)
pred_y = model2.predict(X_test_2)

print("Hesitant: ", round(accuracy_score(y_test, pred_y), 4))


X_train_3 = hstack([X_train_counts, X_train_gram])
X_test_3 = hstack([X_test_counts, X_test_gram])

model3 = LogisticRegression(solver = 'saga', random_state=227)
model3.fit(X_train_3, y_train)
pred_y = model3.predict(X_test_3)

print("Grammatical errors: ", round(accuracy_score(y_test, pred_y), 4))


X_train_4 = hstack([X_train_counts, X_train_reduced])
X_test_4 = hstack([X_test_counts, X_test_reduced])

model4 = LogisticRegression(solver = 'saga', random_state=227)
model4.fit(X_train_1, y_train)
pred_y = model4.predict(X_test_4)

print("Reduced words: ", round(accuracy_score(y_test, pred_y), 4))


X_train_5 = hstack([X_train_counts, X_train_um])
X_test_5 = hstack([X_test_counts, X_test_um])

model5 = LogisticRegression(solver = 'saga', random_state=227)
model5.fit(X_train_5, y_train)
pred_y = model5.predict(X_test_5)

print("Ums: ", round(accuracy_score(y_test, pred_y), 4))

# combining all features
X_train = hstack([X_train_counts, X_train_hesitant, X_train_restart, X_train_gram, X_train_reduced, X_train_um])
X_test = hstack([X_test_counts, X_test_hesitant, X_test_restart, X_test_gram, X_test_reduced, X_test_um])

model6 = LogisticRegression(solver = 'saga', random_state=227)
model6.fit(X_train, y_train)
pred_y = model6.predict(X_test)

print("All: ", round(accuracy_score(y_test, pred_y), 4))

# just the additional features
X_train = np.hstack([np.array(X_train_hesitant), np.array(X_train_restart), np.array(X_train_gram), np.array(X_train_reduced), np.array(X_train_um)])
X_test = np.hstack([np.array(X_test_hesitant), np.array(X_test_restart), np.array(X_test_gram), np.array(X_test_reduced), np.array(X_test_um)])

model7 = LogisticRegression(solver = 'saga', random_state=227)
model7.fit(X_train, y_train)
pred_y = model7.predict(X_test)

print("Just these: ", round(accuracy_score(y_test, pred_y), 4))