import pyprind
import pandas as pd
from string import  punctuation
import re
import os
import numpy as np
from collections import Counter

# Read csv file
from rnn import SentimentRNN

df = pd.read_csv('music_review.csv', encoding='utf-8')

counts = Counter()
pbar = pyprind.ProgBar(len(df['review']), title='Counting words occurrences')

for i, review in enumerate(df['review']):
    text = ''.join([c if c not in punctuation else ' '+c+' ' for c in str(review)]).lower()
    df.loc[i, 'review'] = text
    pbar.update()
    counts.update(text.split())

# Mapping the each unique word into an integer
word_counts = sorted(counts, key=counts.get, reverse=True)
print(word_counts[:5])
word_to_int = {
    word: ii for ii, word in enumerate(word_counts, 1)
}

mapped_reviews = []
pbar = pyprind.ProgBar(len(df['review']), title='Map reviews to integers')
for review in df['review']:
    mapped_reviews.append([word_to_int[word] for word in review.split()])
    pbar.update()


# zero padding process
sequence_length = 200
sequences = np.zeros((len(mapped_reviews), sequence_length), dtype=int)

for i, row in enumerate(mapped_reviews):
    review_arr = np.array(row)
    sequences[i, -len(row):] = review_arr[-sequence_length:]

# split data set into training part and testing part
X_train = sequences[:44705, :]
y_train = df.loc[:44705, 'sentiment'].values

X_test = sequences[44705:, :]
y_test = df.loc[44705:, 'sentiment'].values

np.random.seed(123)


if 'TRAVIS' in os.environ:
    X_train = sequences[:250, :]
    y_train = df.loc[:250, 'sentiment'].values
    X_test = sequences[250:500, :]
    y_test = df.loc[250:500, 'sentiment'].values


n_words = max(list(word_to_int.values())) + 1

rnn = SentimentRNN(n_words=n_words,
                   seq_len=sequence_length,
                   embed_size=256,
                   lstm_size=128,
                   num_layers=1,
                   batch_size=100,
                   learning_rate=0.001)

rnn.train(X_train, y_train, num_epochs=40)
preds = rnn.predict(X_test)
y_true = y_test[:len(preds)]
print('Test acc.: %.3f' % (np.sum(preds==y_true)/len(y_true)))


