import argparse
import pickle
import nltk
import string
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as skl_stopwords

lemmatizer = WordNetLemmatizer()
tokenizer= TreebankWordTokenizer()


# customized module
import generic_functions as gf


def tokenize_lemmatize(text, stopwords=skl_stopwords, tokenizer=tokenizer, lemmatizer=lemmatizer):
    text = ''.join([ch for ch in text if ch not in string.punctuation]) #remove punctuation
    tokens = nltk.word_tokenize(text) 
    return [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords] 

def main():
    # load models
    count_vectorizer = gf.load_model('count_vectorizer.pkl')
    tfid_transformer = gf.load_model('tfid_transformer.pkl')
    naive_bayes = gf.load_model('naive_bayes.pkl')

    # initialize ArgumentParser
    parser=argparse.ArgumentParser()

    # add arguments
    parser.add_argument('-l', '--lyrics', type=str, help='song line to predict')
    args=parser.parse_args()

    # Check if the lyrics input is empty
    if not args.lyrics:
        print('Error: Lyrics input is empty.')
    else:
        # make it iterable to be fit for transform
        iterable_lyrics = [args.lyrics]

        # Transform
        lyrics_transform = tfid_transformer.transform(count_vectorizer.transform(iterable_lyrics))

        # Set feature names
        feature_names = count_vectorizer.get_feature_names_out()

        # Create a DataFrame with the transformed data and feature names
        lyrics_df = pd.DataFrame(lyrics_transform.toarray(), columns=feature_names)
        
        # predict
        artist_pred = naive_bayes.predict(lyrics_df)

        print(f'The singer who sang this is likely {artist_pred[0]}')

if __name__ == '__main__':
    main()