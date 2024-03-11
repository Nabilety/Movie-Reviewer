"""import tarfile
with tarfile.open('aclImdb_v1.tar.gz', 'r:gz') as tar:
    tar.extractall()
"""
import pyprind
import pandas as pd
import numpy as np
import os
import sys

# ## Preprocessing the movie dataset into more convenient format

# Install pyprind by uncommenting the next code cell.

# change the 'basepath' to the directory of the
# unzipped movie dataset
basepath = 'aclImdb'

def convertIMDBFileToCSV(path):
    labels = {'pos': 1, 'neg': 0}
    pbar = pyprind.ProgBar(50000, stream=sys.stdout)
    df = pd.DataFrame()
    for s in ('test', 'train'):
        for l in ('pos', 'neg'):
            path = os.path.join(basepath, s, l)
            for file in sorted(os.listdir(path)):
                with open(os.path.join(path, file),
                          'r', encoding='utf-8') as infile:
                    txt = infile.read()
                df = pd.concat([df, pd.DataFrame([[txt, labels[l]]], columns=['review', 'sentiment'])],
                               ignore_index=True)
                pbar.update()
    # df.columns = ['review', 'sentiment']

    # Shuffling the DataFrame:
    np.random.seed(0)
    df = df.reindex(np.random.permutation(df.index))
    df.to_csv('movie_data.csv', index=False, encoding='utf-8')

    # Optional: Saving the assembled data as CSV file:
    df = pd.read_csv('movie_data.csv', encoding='utf-8')
    # following column renaming is necessary on some computers:
    df = df.rename(columns={"0": "review", "1": "sentiment"})
    print(df.head(3))

    return df


# Call the function to check if the CSV file exists
if not os.path.exists('movie_data.csv'):
    df = convertIMDBFileToCSV(basepath)
else:
    print(f"CSV file 'movie_data.csv' already exists. Skipping function execution")
    df = pd.read_csv('movie_data.csv', encoding='utf-8')


# sanity check that the DataFrame contains all 50000 rows
print(df.head(3))
print(df.shape)


# # Introducing the bag-of-words model

# ...

# ## Transforming documents into feature vectors

# By calling the fit_transform method on CountVectorizer, we just constructed the vocabulary of the bag-of-words model
# and transformed the following three sentences into sparse feature vectors:
# 1. The sun is shining
# 2. The weather is sweet
# 3. The sun is shining, the weather is sweet, and one and one is two
# Transforming words into feature vectors

from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()
docs = np.array(['The sun is shining',
                 'The weather is sweet',
                 'The sun is shining, the weather is sweet, and one and one is two'
                 ])
bag = count.fit_transform(docs)

# Now let us print the contents of the vocabulary
# to get a better understanding of the underlying concepts:
print(count.vocabulary_)

# As we can see from executing the preceding command, the vocabulary is stored in a Python dictionary,
# which maps the unique words that are mapped to integer indices. Next let us print the feature vectors that we just created:
print(count.get_feature_names_out()) # the column indices in this will represent the column features in the matrix below
print(bag.toarray()) # each row represents the sentences we constructed in docs

# Each index position in the feature vectors shown here corresponds
# to the integer values that are stored as dictionary items in the CountVectorizer vocabulary.
# For example, the  rst feature at index position 0 resembles the count of the word and,
# which only occurs in the last document, and the word is at
# index position 1 (the 2nd feature in the document vectors) occurs in all three sentences.
# Those values in the feature vectors are also called the raw term frequencies:
# *tf (t,d)*—the number of times a term t occurs in a document *d*.

# ## Assessing word relevancy via term frequency-inverse document frequency
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer(use_idf=True,
                         norm='l2',
                         smooth_idf=True)

np.set_printoptions(precision=2)
#TfidfTransformer class normalizes the tf-idfs directly.
# But good practice would be to normalize raw term frequencies before calculating the tf-idfs
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())


# ## Cleaning text data
import re
print(df.loc[0, 'review'][-50:])

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text

preprocessor(df.loc[0, 'review'][-50:])
print(preprocessor("</a>This :) is :( a test :-)!"))

# apply preprocessor function to all movie reviews in our dataframe:
df['review'] = df['review'].apply(preprocessor)


# ## Processing documents into tokens
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords

# tokenize by splitting individual words in a document at their whitespace characters
def tokenizer(text):
    return text.split()
print(tokenizer('runners like running and thus they run'))

# word stemming using Porter stemming algorithm, reducing words to their root form i.e. runners -> runner
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
print(tokenizer_porter('runners like running and thus they run'))
# Porter stemming is the oldest/simplest algorithm. Other popular stemming alogirhtms include Snowball stemmer,
# Lancaster stemmer (Paice/Husk stemmer) available through the NLTK package.

# while stemming can create non-real words such as 'thu' from 'thus, we can use lemmatization to obtain canonical forms
# of individual words (correct grammatic) - using lemmas. However lemmatization is computationally more difficult and expensive



# stop word removal (removal of words extremely commonly used, i.e. and, is, has, or like).
# this will help with raw or normalized term frequencies rather than tf-idfs, which already downweights frequent words.
# to remove stop words, we will use the set of 127 english stop words available in the NLTK library by calling nltk.download
nltk.download('stopwords')
stop = stopwords.words('english')
print([w for w in tokenizer_porter('a runner likes running and runs a lot') if w not in stop])


"""
# # Training a logistic regression model for document classification

# Strip HTML and punctuation to speed up the GridSearch later:
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[:25000, 'review'].values
y_test = df.loc[:25000, 'sentiment'].values

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
# TfidfVectorizer need token_pattern=None, when we define our own tokenizer. To prevent the warnings
tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None,
                        token_pattern=None)

small_param_grid = [
    {
        'vect__ngram_range': [(1, 1)],
        'vect__stop_words': [None],
        'vect__tokenizer': [tokenizer, tokenizer_porter],
        'clf__penalty': ['l2'],
        'clf__C': [1.0, 10.0]
    },
    {
        'vect__ngram_range': [(1, 1)],
        'vect__stop_words': [stop, None],
        'vect__tokenizer': [tokenizer],
        'vect__use_idf': [False],
        'vect__norm': [None],
        'clf__penalty': ['l2'],
        'clf__C': [1.0, 10.0]
    }
]

lr_tfidf = Pipeline([('vect', tfidf), ('clf', LogisticRegression(solver='liblinear'))]) #liblinear perform better for large dataset
gs_lr_tfidf = GridSearchCV(lr_tfidf, small_param_grid,
                           scoring='accuracy', cv=5,
                           verbose=2, n_jobs=-1)

# **Important Note about `n_jobs`**
#
# Please note that it is highly recommended to use `n_jobs=-1` (instead of `n_jobs=1`)
# in the previous code example to utilize all available cores on your machine and speed up the grid search.
# However, some Windows users reported issues when running the previous code with the `n_jobs=-1`
# setting related to pickling the tokenizer and tokenizer_porter functions for multiprocessing on Windows.
# Another workaround would be to replace those two functions, `[tokenizer, tokenizer_porter]`, with `[str.split]`.
# However, note that the replacement by the simple `str.split` would not support stemming.
gs_lr_tfidf.fit(X_train, y_train)

print(f'Best parameter set: {gs_lr_tfidf.best_params_}')
print(f'CV Accuracy: {gs_lr_tfidf.best_score_:.3f}')

clf = gs_lr_tfidf.best_estimator_
print(f'Test Accuracy: {clf.score(X_test, y_test):.3f}')
"""

# # Working with bigger data - online algorithms and out-of-core learning
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import pyprind
stop = stopwords.words('english')


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv) # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

print(next(stream_docs(path='movie_data.csv')))

def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

# NOTE choosing a large number of features in the HashingVectorizer reduces chance of causing hash collisions,
# but we also increase the number of coefficients in our logistic regression model.
vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)
clf = SGDClassifier(loss='log_loss', random_state=1)
doc_stream = stream_docs(path='movie_data.csv')

# Having set up all complementary function we can now starte the out-of-core learning.
# Note we once again use pyprind to estimate progress of our learning algorithm:
# Initialized progress bar object with 45 iterations, as we inside the for-loop will iterate over 45 mini-batches of
# documents, where each mini-batch consist of 1000 documents. Once completing learning process, we use 5000 documents to evaluate the model
pbar = pyprind.ProgBar(45)
classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()

X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print(f'Accuracy: {clf.score(X_test, y_test):.3f}')

# use last 5000 documents to update our model:
clf = clf.partial_fit(X_test, y_test)


# ## Topic modeling

# ### Decomposing text documents with Latent Dirichlet Allocation

# ### Latent Dirichlet Allocation with scikit-learn
df = pd.read_csv('movie_data.csv', encoding='utf-8')
# the following is necessary on some computers:
df = df.rename(columns={"0": "review", "1": "sentiment"})

# create bag-of-words matrix as input to the LDA with CountVectorizer
# use english stop word library:
count = CountVectorizer(stop_words="english",
                        max_df=.1,
                        max_features=5000)
X = count.fit_transform(df['review'].values)
# note  we put maximum document frequency of words to be considered to 10% (max_df=.1) to exclude frequent words across documents.
# the idea is words that occur frequently are less likely associated with speicfic topics.
# to limit the dimensionality of the dataset to improve the inference performed by LDA, we limited the numbers of words
# to be considered to the most frequently occuring 5000 words (max_features=5000)
# Both hyperparameter values are chosen arbitrarily, and should be tuned when comparing results in practice
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=10,
                                random_state=123,
                                learning_method='batch')
X_topics = lda.fit_transform(X)
# setting learning_method='batch',
# we let the lda do its estimation based on all available training data (bag-of-words matrix) in one iteration,
# which is slower than the alternative 'online' (online/mini-batch) learning method, but can lead to more accurate results.

# print component attribute of lda instance, storing matrix containing a word importance (here 5000) for each of 10 topics
print(lda.components_.shape)

# print five most important words for each 10 topics. Note word importance values are ranked in increasing order
# so in other to print the top five words, we need to sort the topic array in reverse order:
n_top_words = 5
feature_names = count.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    print(f'Topic {(topic_idx + 1)}:')
    print(' '.join([feature_names[i]
                    for i in topic.argsort() \
                        [:-n_top_words - 1:-1]]))


# To confirm that the categories make sense based on the reviews, let's plot 5 movies from the horror movie category
# (category 6 at index position 5):
horror = X_topics[:, 5].argsort()[::-1]
for iter_idx, movie_idx in enumerate(horror[:3]):
    print(f'\nHorror movie #{(iter_idx + 1)}:')
    print(df['review'][movie_idx][:300], '...')











