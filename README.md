# text-classification
In this project, a text classification model is built based on song lyrics. The task is to predict the artist from a piece of text.

## Pre-processing:
1. Tokenisation
2. Clean the text (capitalization, punctuations)
3. Stemming - the reduction of the word to its (pseudo)stem by removing suffixes via some heuristic rules. Does not always result in a real word at the end
4. Lemmatisation - the conversion of a word to its dictionary form
5. Removing stopwords
6. Vectorization

#### CountVectorizer
- Remove list of stopwords
- Remove punctuation marks
- Remove the words that appear in more than X% of documents

## Models:
1. RandomForestClassifier
2. Naive Bayes Classifier
