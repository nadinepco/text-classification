import random, re, string
from bs4 import BeautifulSoup
import nltk
import pickle
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as skl_stopwords

def get_header():
    """Gets a random user agent and returns it as a header
    
    :return: header
    :rtype: dict"""
    
    agent_list = list()
    
    # list of browsers (each browser has a text file with UA)
    browser_list = ['Firefox','Internet+Explorer','Opera','Safari','Chrome','Edge','Android+Webkit+Browser']

    # randomly select which file browser to read
    browser = browser_list[random.randint(0,len(browser_list)-1)]
    
    # file path
    file_path = 'user_agents/'+browser+'.txt'
    
    with open(file_path, 'r') as file:
        for line in file:
            if not line.startswith('More'):
                # Remove newline character and leading/trailing whitespace
                cleaned_line = line.strip()
                agent_list.append(cleaned_line)
    user_agent = random.choice(agent_list)
    header={f'User-Agent': user_agent}
    return header

def write_file(text,filename):
    """Writes to a text file
        :param text: text to be written to the file
        :param filename: name of the file"""
    with open(filename, "w") as f:
        f.write(text)
        
    
def read_file(filename):
    """Reads from a file
        :param filename: name of the file to be read
        :return: contents of the file
        :rtype: string"""
    with open(filename, 'r') as f:
        html_doc = f.read()
    return html_doc

# have the titles cleaned since there are duplicate titles that have feat. or remix
# e.g. Girls Like You [feat._Cardi_B] and Girls Like You
def format_title(title):
    """Used to format the title by doing the following:
    1. Getting the substring of the title before "[" 
    2. Set the format to title()
    3. Replace space with underscore
    
    :param title: name of the title to be formatted"""
    
    title = title.split('[')[0].split('(')[0].strip()
    
    # replace / with ()
    title = title.replace('/','(')+')' if ('/' in title) else title
    
    # replace space with underscore
    title = string.capwords(title).replace(' ','_')
    
    return title

def get_lyrics(filename):
    """ Parse the file and return the text in the lyric-body-text id
    :return: lyrics section
    :rtype: string
    """
    html_doc = read_file(filename)
    soup = BeautifulSoup(markup=html_doc, features='html.parser')
    lyrics = soup.find('pre', attrs = {'id':'lyric-body-text'})

    if lyrics is None:
        return lyrics
    else:
        return lyrics.text


def save_model(filename,model):
    """ Save model in a file 
    : param filename: assign name of the file to be saved
    : param model: model to be saved
    """
    with open(f'{filename}.pkl', 'wb') as file:
        pickle.dump(model, file)
        
def load_model(filename):
    """ Load model from a file
    : param filename: name of the file where the model is saved
    """
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

def clean_dataframe(df):
    # Split lyrics with "\n"
    df['lyrics'] = df['lyrics'].str.split('\n')
    
    # Explode the lyrics column into multiple rows
    df = df.explode('lyrics').reset_index(drop=True)
    
    # drop columns link and title
    df.drop(columns=['link','title'], axis=1, inplace=True)    
    
    # remove _ in artist column and capitalize names
    df['artist'] = df['artist'].str.replace('_', ' ').apply(string.capwords)   
    
    # set artist as index
    df = df.set_index('artist')       
    return df