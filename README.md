# PoliticalNLP
Determining Political Affiliation of Poster on Thread Title (Reddit)

## Problem: 
It is more and more common that people self-segregate into bubbles in social spaces especially based off political affiliation. I wanted to determine if we could easily distinguish how these groups are talking and what they are talking about.

## Objective: 
Create a model that can accurately predict Political Affiliation of a poster based on the content of what they posted. 

## Methodology: 
Used an NLP (Natural Language Processing) Model to predict political affiliation based off of a datasource of 21.5k Reddit Thread Titles

## Data
For this exploration I used a datasource of 13k Reddit titles from Kaggle. In addition, I pulled an additional 10k posts from the Reddit API to supplement my model.

The data contained:
1. Post Title
2. Political Affiliation
3. Number of Upvotes
4. Number of Comments
   
An assumption of this model is that if a person posts on a certain community they belong to that community. This isn’t always true, but is good enough for our purposes 

## EDA 
The data overall was relatively ‘clean’, though it did tend to have quite a few posts that were very short like “yes”, “yup”, “excellent”, etc. 
I opted to keep posts with a length of at least 15 characters as this had good results.

There was also an imbalance in the data toward ‘liberal’ posts as opposed to conservative ones. I oversampled the minority class to address this.

![plot](images/Length_Histogram.png)

## Data Preparation & Data Engineering

There was some preprocessing involved in getting ready for classification:
1. Lowercasing
2. Unpunctuating
3. Tokenizing (e.g. breaking up into words)
4. Lemmatizing (e.g. happiness and happy -> happy)
5. Word2Vec Vectorizing (to convert words into arrays of numbers for the model to use)
   
I iterated through numerous new variables as well, but landed on the following as being predictive:
1. Number of proper nouns
2. Length of the sentence
3. Number of capitalized words

## Base Model (Logistic)

As a baseline I created a simple Logistic Regression. This model performed poorly, but did do better than chance.
Accuracy: 65%

The model struggled with over-predicting the conservative class, doing a better job with conservative posters.

![plot](images/Confusion_Logistic_Model.png)

## Final Model (LSTM)

My final model was an LSTM model that had two branches. One for the vectorized text and the other for additional variables.

I did try this with both vectorized sentences and vectorized words and their performance was similar, but sentences slightly edged out words.
Loss: Binary Cross-entropy
Optimizer: Adam
Epochs: 104 (early dropout on 150)
LSTM Units (256 / 128)
Dense Units( 64)
Dropout (0.05)

![plot](images/LSTM.png)

The model performed very well with an accuracy of 85% and an ROC-AUC of 91%. 

![plot](images/Confusion_Sentence_Model.png)

There was some overfitting, but not anything out of hand. Around about 20 epochs training and validation diverge, but within reasonable levels.

![plot](images/Epochs_Sentence_Model.png)

The model had a high accuracy but did tend to fail when text was short and/or vague or when it represented news that could easily have been posted by either group. Where it performed best, text was medium length, punchy and contained keywords that were easy to identify  

## Other Analysis

I also attempted to use sentiment analysis to see how it correlated with political affiliation using VADER (Valence Aware Dictionary for Sentiment Reasoning). 
However, this did not have great results as the corpus contains a lot of tricky language that it couldn’t tease apart (e.g. quoting someone speaking negatively and then saying you like it)

Similarly, despite the relatively small size of the corpus, the number of topics swiftly ballooned making it difficult to analyze. 
Certain topics were easy to identify like the Supreme Court or Jan 6 Insurrection, but others were elusive like abortion vs. Ron Desantis (sometimes about abortion, but not always)

## For the Future
1. Create a custom stopword dictionary to eliminate redundant words without losing information
2. Continue pulling data to expand the corpus
3. Explore other modeling solutions (e.g. BERT)
4. Expanding Topic Analysis (how many topics are there?)
5. How well does this perform on other political text? Could it be used on Facebook posts for example?

