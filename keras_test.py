# -*- coding: utf-8 -*-

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import SGD
from keras.utils import to_categorical
from scipy.sparse import linalg

xdata = []
ydata = []

file = open('Digital_Music_5_10000.json', 'r')#for time and performance issues, I used only the first 10,000 reviews

for l in file:
    temp = eval(l)["reviewText"]
    temp1 = eval(l)["overall"]
    xdata.append(temp)
    if temp1 < 3:# convert the rating into 3 categories: positive, neutural, negative
        ydata.append(-1)
    elif temp1 == 3:
        ydata.append(0)
    else:
        ydata.append(1)


# represent the raw data in tf-idf form
vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'\b\w{2,}\b', min_df=1, max_df=0.1)
data = vectorizer.fit_transform(xdata)
#building neural network
model = Sequential()
model.add(Dense(64, input_dim=len(data.toarray()[0]), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
#train with the network
model.fit(data[:8000].toarray(), ydata[:8000], epochs=5, batch_size=128)
#evaluate model
loss_and_metrics = model.evaluate(data[8000:].toarray(), ydata[8000:], batch_size=128)
print(model.metrics_names)
print(loss_and_metrics)
