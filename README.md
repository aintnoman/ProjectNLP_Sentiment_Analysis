# ProjectNLP_Sentiment_Analysis
Sentiment Analysis Based On Deeping Learning from Amazon Product Review

#### Setup:
1. Please make sure you have install following libraries before you run the applications:
    - `Jupiter Notebook`
    -  `NLTK`
2. For reason of license of training data, please [download](http://jmcauley.ucsd.edu/data/amazon/) `music_reviews.json` and put it in with `.ipynb` in the same directory.
#### Processed Data:
1. After running `process_data.ipynb`, you will have two txt files: `review_list.txt` and `overall_list.txt`.
2. `review_list.txt`: Includes tokenized, lemmatized, stop_word processed data.
3. `overall_list.txt`: Labels for each training review sample. 1: Pos, 0: Neg.
4. `music_review.csv`: Processed data formatted in the csv file for training model later.
#### Training Via RNN
Excute `train.py` and find the trained model in the 'model' folder
#### Accuracy Test
Excute `accuracy_test.ipynb` by Jupyter Notebook after trained the model.

