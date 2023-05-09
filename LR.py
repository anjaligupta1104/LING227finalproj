import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import re


path_dict = {"./data/not_impaired": 0, "./data/impaired": 1}
data = []

for path, label in path_dict.items():
    for i in os.listdir(path):
        if i.endswith('.txt'):
            j = pd.read_csv(os.path.join(path, i), sep = '\t', header = None)
            j['sli'] = label
            data.append(j)

df = pd.concat(data, ignore_index=True)
x = df.drop('sli', axis = 1)
y = df['sli']

print(df)
print(x.shape)
print(x[:5])

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.35, random_state=227, stratify=y)

print(train_x[0])

# No Features
v = TfidfVectorizer(stop_words='english')
x_tr = v.fit_transform(train_x[0])
x_t = v.transform(test_x[0])

model = LogisticRegression(solver = 'lbfgs', random_state=227, max_iter= 15000)
model.fit(x_tr, train_y)
pred_y = model.predict(x_t)

print(accuracy_score(test_y, pred_y))

# First Feature
data = []

for path, label in path_dict.items():
    for i in os.listdir(path):
        if i.endswith('.txt'):
            j = pd.read_csv(os.path.join(path, i), sep = '\t', header = None)
            j['sli'] = label
            data.append(j)

df = pd.concat(data, ignore_index=True)

karets_exp = re.compile(r'<[^>]*>')
df['restarted_phrases'] = df[0].apply(lambda x: 1 if karets_exp.search(x) else 0)

x = df.drop('sli', axis = 1)
y = df['sli']

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.35, random_state=227, stratify=y)
v = TfidfVectorizer(stop_words='english')
x_tr = v.fit_transform(train_x[0])
x_t = v.transform(test_x[0])

model = LogisticRegression(solver = 'lbfgs', random_state=227, max_iter= 15000)
model.fit(x_tr, train_y)
pred_y = model.predict(x_t)

print(accuracy_score(test_y, pred_y))

# Second Feature
data = []

for path, label in path_dict.items():
    for i in os.listdir(path):
        if i.endswith('.txt'):
            j = pd.read_csv(os.path.join(path, i), sep = '\t', header = None)
            j['sli'] = label
            data.append(j)

df = pd.concat(data, ignore_index=True)

df['hesitant'] = df[0].apply(lambda x: 1 if '&-' in x else 0)
karets_exp = re.compile(r'<[^>]*>')
df['restarted_phrases'] = df[0].apply(lambda x: 1 if karets_exp.search(x) else 0)

x = df.drop('sli', axis = 1)
y = df['sli']

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.35, random_state=227, stratify=y)
v = TfidfVectorizer(stop_words='english')
x_tr = v.fit_transform(train_x[0])
x_t = v.transform(test_x[0])

model = LogisticRegression(solver = 'lbfgs', random_state=227, max_iter= 15000)
model.fit(x_tr, train_y)
pred_y = model.predict(x_t)

print(accuracy_score(test_y, pred_y))

# Third Feature 
data = []

for path, label in path_dict.items():
    for i in os.listdir(path):
        if i.endswith('.txt'):
            j = pd.read_csv(os.path.join(path, i), sep = '\t', header = None)
            j['sli'] = label
            data.append(j)

df = pd.concat(data, ignore_index=True)

df['reduced_words'] =  df[0].apply(lambda x: 1 if '&~' in x else 0)
df['hesitant'] = df[0].apply(lambda x: 1 if '&-' in x else 0)
karets_exp = re.compile(r'<[^>]*>')
df['restarted_phrases'] = df[0].apply(lambda x: 1 if karets_exp.search(x) else 0)

x = df.drop('sli', axis = 1)
y = df['sli']

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.35, random_state=227, stratify=y)
v = TfidfVectorizer(stop_words='english')
x_tr = v.fit_transform(train_x[0])
x_t = v.transform(test_x[0])

model = LogisticRegression(solver = 'lbfgs', random_state=227, max_iter= 15000)
model.fit(x_tr, train_y)
pred_y = model.predict(x_t)

print(accuracy_score(test_y, pred_y))

# Fourth Feature
data = []

for path, label in path_dict.items():
    for i in os.listdir(path):
        if i.endswith('.txt'):
            j = pd.read_csv(os.path.join(path, i), sep = '\t', header = None)
            j['sli'] = label
            data.append(j)

df = pd.concat(data, ignore_index=True)

df['um'] = df[0].apply(lambda x: x.count('um'))
df['reduced_words'] =  df[0].apply(lambda x: 1 if '&~' in x else 0)
df['hesitant'] = df[0].apply(lambda x: 1 if '&-' in x else 0)
karets_exp = re.compile(r'<[^>]*>')
df['restarted_phrases'] = df[0].apply(lambda x: 1 if karets_exp.search(x) else 0)

x = df.drop('sli', axis = 1)
y = df['sli']

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.35, random_state=227, stratify=y)
v = TfidfVectorizer(stop_words='english')
x_tr = v.fit_transform(train_x[0])
x_t = v.transform(test_x[0])

model = LogisticRegression(solver = 'lbfgs', random_state=227, max_iter= 15000)
model.fit(x_tr, train_y)
pred_y = model.predict(x_t)

print(accuracy_score(test_y, pred_y))

# Fifth Feature
data = []

for path, label in path_dict.items():
    for i in os.listdir(path):
        if i.endswith('.txt'):
            j = pd.read_csv(os.path.join(path, i), sep = '\t', header = None)
            j['sli'] = label
            data.append(j)

df = pd.concat(data, ignore_index=True)

df['grammatical_error'] = df[0].apply(lambda x: 1 if '[*]' in x else 0)
df['um'] = df[0].apply(lambda x: x.count('um'))
df['reduced_words'] =  df[0].apply(lambda x: 1 if '&~' in x else 0)
df['hesitant'] = df[0].apply(lambda x: 1 if '&-' in x else 0)
karets_exp = re.compile(r'<[^>]*>')
df['restarted_phrases'] = df[0].apply(lambda x: 1 if karets_exp.search(x) else 0)

x = df.drop('sli', axis = 1)
y = df['sli']

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.35, random_state=227, stratify=y)
v = TfidfVectorizer(stop_words='english')
x_tr = v.fit_transform(train_x[0])
x_t = v.transform(test_x[0])

model = LogisticRegression(solver = 'lbfgs', random_state=227, max_iter= 15000)
model.fit(x_tr, train_y)
pred_y = model.predict(x_t)

print(accuracy_score(test_y, pred_y))