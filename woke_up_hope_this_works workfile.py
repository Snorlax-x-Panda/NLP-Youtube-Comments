# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 11:38:35 2023

@author: james
"""


#   1. Load the data into a pandas data frame.
import pandas as pd

data = pd.read_csv("C:/Users/james/OneDrive/Desktop/School Winter 2023/AI/NLP Assignment/Youtube01-Psy.csv")



#   2. Carry out some basic data exploration and present your results. 
#   (Note: You only need two columns for this project, make sure you 
#   identify them correctly, if any doubts ask your professor)

data.shape
data.describe()
data.columns
data['CLASS'].value_counts()

# Identifying the two columns needed. [ Class, Content ]
final_cols = ['CONTENT', 'CLASS' ]
data_final = data[final_cols]
data_content = data['CONTENT']



#   3. Using nltk toolkit classes and methods prepare the data for model 
#   building, refer to the third lab tutorial in module 11 (Building a
#   Category text predictor ). Use count_vectorizer.fit_transform().
import nltk
from sklearn.feature_extraction.text import CountVectorizer

cnt_vectorizer = CountVectorizer(lowercase=True, token_pattern="[^\W\d_]+", stop_words='english') 

# Word Count
word_count = cnt_vectorizer.fit_transform(data_final.CONTENT)

#   4. Present highlights of the output (initial features) such as the new 
#   shape of the data and any other useful information before proceeding.
word_count.shape
word_count.dtype
feature_name = cnt_vectorizer.get_feature_names_out()
"""
Contains the 350 rows of content and 1418 words. The word count is described
by numbers and the amount counted.
"""

#   5. Downscale the transformed data using tf-idf and again present 
#   highlights of the output (final features) such as the new shape 
#   of the data and any other useful information before proceeding.
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(word_count)

# convert back to pd
train_tfidf_df = pd.DataFrame(train_tfidf.toarray(), columns=cnt_vectorizer.get_feature_names_out())
# join y column before splitting
y_col = data_final.CLASS
train_tfidf_df = train_tfidf_df.join(y_col)

"""
"""

#   6. Use pandas.sample to shuffle the dataset, set frac = 1.
#train_tfidf_final = train_tfidf.sample(frac=1)

data_final_shuffle = train_tfidf_df.sample(frac=1)
''''''

#   7. Using pandas split your dataset into 75% for training and 
#   25% for testing, make sure to separate the class from the feature(s). 
#   (Do not use test_train_ split)

split_percent = 0.75
split_mark = int(len(data_final_shuffle) * split_percent)

# split y from shuffled df
y_shuffle = data_final_shuffle.loc[:, 'CLASS']
y_train = y_shuffle[:split_mark]
y_test = y_shuffle[split_mark:]
#data_final_shuffle.drop(columns="CLASS")

# split x
x_train = data_final_shuffle.drop(columns="CLASS")[:split_mark]
x_test = data_final_shuffle.drop(columns="CLASS")[split_mark:]


'''
x_train = train.iloc[: ,0]
y_train = train.iloc[: ,1]
x_test = test.iloc[: ,0]
y_test = test.iloc[: ,1]    
'''

#   8. Fit the training data into a Naive Bayes classifier. 
'''from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train, y_train)'''

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train, y_train)

#   9. Cross validate the model on the training data using 5-fold and print 
#   the mean results of model accuracy.
from sklearn.model_selection import cross_val_score
acc_scores = cross_val_score(model, x_train, y_train, cv=5)
print(acc_scores.mean())

#   10. Test the model on the test data, print the confusion matrix and the 
#   accuracy of the model.
from sklearn.metrics import accuracy_score, confusion_matrix
ypred = model.predict(x_test) 
accuracy_score(y_test, ypred)
confusion_matrix(y_test, ypred)

#   11. As a group come up with 6 new comments (4 comments should be non 
#   spam and 2 comment spam) and pass them to the classifier 
#   and check the results.
test_data = [
        'BUY THIS SONG FOR $3.99 ON APPLE MUSIC.',
        'That dance is worth a million views.',
        'General Tzu was a military general that was famous for his tactics and  teachings.',
        'I really like chocolate bars.',
        'How is this song one of the biggest songs in the world?',
        'Who is the girl dancing with PSY?',
        'Did you know that a person defecates immediately after death?',
        'Hello, thius iassasd l is jawal !!!'
    ]

# Vectorize Data
test_cnt = cnt_vectorizer.transform(test_data)

# Convert to freq values
test_tfidf = tfidf.transform(test_cnt)

# Make prediction with model
predictions = model.predict(test_tfidf)


for data, predict in zip(test_data, predictions):
    print('\nTest Data --> ', data, '\nPrediction --> ', predict)
