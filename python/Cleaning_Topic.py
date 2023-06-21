def lowercase(text):
    return text.lower()

def unpunctuate(text):
    import string
    text = "".join([char for char in text if char not in string.punctuation])
    return text

def remove_stopwords(tokenized_text):
    import nltk
    from nltk.corpus import stopwords

    ENGstopwords = stopwords.words('english')
    list = ['texas','florida','u','come','biden','trump',"it’s",'joe','must','got','know','fact','guy','say','may','think','like','v','-','really','next','new','would','month','year','get','tell','arizona','good','thing', '–', 'way', 'im','make','go','ever','2','1','3','4','5','7','8','9','0','real','talk','conservative','liberal','never','country','us','need','dont','still','great','try','news','man',
           'let','take','help','interview','much','point','keep','even','question','whats',"isn't",'video','see','party','political','government']
    for word in list:
        ENGstopwords.append(word)
    text = [word for word in tokenized_text if word not in ENGstopwords]
    return text

def tokenize(text):
    tokens = text.split()
    return tokens
  
def lemmatize(text):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer() 
    lemtext = []
    for word in text: 
        lemword = lemmatizer.lemmatize(word)
        lemtext.append(lemword)
    return lemtext


def count_capitalized_words(text):
    import re
    capitalized_words = re.findall(r'\b[A-Z][a-zA-Z]+\b', text)
    return len(capitalized_words)

def count_punctuations(text):
    import re
    punctuation = re.findall(r"[!\"#\$%&\'\(\)\*\+,-\./:;<=>\?@\[\\\]\^_`{\|}~]", text)
    return len(punctuation)

def cleaning_and_prep(df):
    import numpy as np
    import pandas as pd
    from imblearn.over_sampling import RandomOverSampler
    from sklearn.utils import shuffle
    
    #Create simple dataframe with only text and predictor
    df = df[['Title', 'Political Lean']] 

    #dummy code predictor
    df['Political Lean'] = df['Political Lean'].replace('Liberal',1)
    df['Political Lean'] = df['Political Lean'].replace('Conservative',0)

    df = df.rename({'Political Lean':'y'},axis=1) #replace column name for simplicity

    df['Length'] = df['Title'].apply(lambda x: len(x))
    df = df[df['Length']>=15]
    
    #Split into X & y
    X = df.drop(['y'],axis=1)
    y = df['y']
    
    #oversample minority class
    oversample = RandomOverSampler(sampling_strategy='minority', random_state=13)
    X, y = oversample.fit_resample(X, y)
    
    #shuffle dataset
    
    X, y = shuffle(X,y, random_state=13)

    #apply lowercase, unpunctuate, tokenize words and then lemmatize
    X['Title'] = X['Title'].apply(lowercase)
    X['Title'] = X['Title'].apply(unpunctuate)
    X['Title'] = X['Title'].apply(tokenize)
    X['Title'] = X['Title'].apply(remove_stopwords)
    X['Title'] = X['Title'].apply(lemmatize)
    X['Title'] = X['Title'].apply(remove_stopwords)

    return X, y

    