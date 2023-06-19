def lowercase(text):
    return text.lower()

def unpunctuate(text):
    import string
    text = "".join([char for char in text if char not in string.punctuation])
    return text

def remove_stopwords(tokenized_text):
    import nltk
    nltk.download('stopwords')

    from nltk.corpus import stopwords

    ENGstopwords = stopwords.words('english')
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

def create_other_var(df):
    #score(upvotes) & number of comments
    df_other_var = df[['Title','Score','Num of Comments']]
    df_other_var = df_other_var.rename({'Score':'Upvotes','Num of Comments':'Comments'},axis=1)
    
    #variable with length of characters
    df_other_var['Length'] = df_other_var['Title'].apply(lambda x: len(x))
    
    #variable with # of capitalized words and punctuation
    df_other_var['Capitals'] = df_other_var['Title'].apply(count_capitalized_words)
    
    #doesn't improve performance so commented out
    #df_other_var['Punctuation'] = df_other_var['Title'].apply(count_punctuations)
    
    df_other_var = df_other_var.drop('Title',axis=1)
    return df_other_var

def nouns(x):
    nouns = []
    for (word,pos) in x:
        if pos.startswith("NN"):
            nouns.append(word)
        number = len(nouns)
    return number

def proper_nouns(x):
    proper_nouns=[]
    for (word,pos) in x:
        if pos.startswith("NNP"):
            proper_nouns.append(word)
        number = len(proper_nouns)
    return number

def verbs(x):
    verbs = []
    for (word,pos) in x:
        if pos.startswith("V"):
            verbs.append(word)
        number = len(verbs)
    return number

def adjectives(x):
    adjectives = []
    for (word,pos) in x:
        if pos.startswith("JJ"):
            adjectives.append(word)
        number = len(adjectives)
    return number

def pos_tag(df_token):
    from nltk import pos_tag
    df_token['POS'] = df_token['Title'].apply(pos_tag)
    
    #commented out those that didn't improve model
    #df_token['Nouns'] = df_token['POS'].apply(nouns)
    df_token['Proper_Nouns'] = df_token['POS'].apply(proper_nouns)
    #df_token['Verbs'] = df_token['POS'].apply(verbs)
    #df_token['Adjectives'] = df_token['POS'].apply(adjectives)
    
    df_token = df_token.drop('POS',axis=1)
    return df_token


def cleaning_and_prep(df):
    ''' adding a dataframe to this will clean it and do data preparation 
    on the text column 'Title' and output a dataframe ready for inputting into LSTM model
    requires lowercase, unpunctuate, tokenize, get_vector and lemmatize functions
    
    example: X_train, X_test, y_train, y_test = cleaning_and_prep(df)
    '''
    import numpy as np
    import pandas as pd
    from imblearn.over_sampling import RandomOverSampler
    from sklearn.utils import shuffle
    from sklearn.model_selection import train_test_split
    from gensim.models import Word2Vec
    
    #create extra variables for later
    df_other_var = create_other_var(df)
    
    #Create simple dataframe with only text and predictor
    df = df[['Title','Political Lean']] 

    #dummy code predictor
    df['Political Lean'] = df['Political Lean'].replace('Liberal',1)
    df['Political Lean'] = df['Political Lean'].replace('Conservative',0)

    df = df.rename({'Political Lean':'y'},axis=1) #replace column name for simplicity
    
    df = pd.concat([df,df_other_var],axis=1)
    
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
    X['Title'] = X['Title'].apply(lemmatize)
    
    #create a word2vec of the frame
    all_words = X['Title'].tolist()
    word2vec = Word2Vec(all_words)
    
    #ineffective so commenting out
    #X['Word'] = X['Title'].apply(lambda x: len(x))
    
    X = pos_tag(X)
    
    def get_vector(word_list):
        from gensim.models import Word2Vec
        import numpy as np

        vector = np.zeros(word2vec.vector_size)  # Create an empty vector of appropriate size
        count = 0  # Initialize a count variable
        for word in word_list:
            if word in word2vec.wv:
                vector += word2vec.wv[word]  # Add the word vector to the overall vector
                count += 1  # Increment the count
        if count != 0:
            vector /= count  # Divide the vector by the count to get the average
        return vector
    
    X['Title'] = X['Title'].apply(get_vector)
    
    # Convert the vector column to a list of lists
    vector_list = X['Title'].tolist()

    # Convert the list of lists to a NumPy array with float32 data type
    vector_array = np.array(vector_list, dtype=np.float32)

    word_vectors = word2vec.wv.vectors
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.15, random_state = 13)
    return X_train, X_test, y_train, y_test

def stack_vectors(X_train,X_test):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    
    X_train_vec = np.stack(X_train['Title'].values)
    X_test_vec = np.stack(X_test['Title'].values)
    
    X_train_vec = np.reshape(X_train_vec, (X_train_vec.shape[0], 1, 100))
    X_test_vec = np.reshape(X_test_vec, (X_test_vec.shape[0], 1, 100))
    
    scaler = StandardScaler()
    X_train = X_train.drop('Title',axis=1)
    X_train = scaler.fit_transform(X_train)
    
    X_test = X_test.drop('Title',axis=1)
    X_test = scaler.transform(X_test)
    
    return X_train_vec, X_test_vec, X_train, X_test, scaler
    