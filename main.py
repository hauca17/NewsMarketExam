# Required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import seaborn as sns
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
from pprint import pprint
import pyLDAvis.gensim


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Read CSV file into a data frame
df = pd.read_csv('reddit_wsb.csv')

# First 5 rows
df.head()

# Information about data types and memory usage
df.info()

# Remove columns from the dataframe
df.drop('id', axis=1, inplace=True)
df.drop('url', axis=1, inplace=True)
df.drop('comms_num', axis=1, inplace=True)
df.drop('created', axis=1, inplace=True)
df.drop('timestamp', axis=1, inplace=True)
df.drop('score', axis=1, inplace=True)

# Check for any missing value
df.isna().sum()

# Check for any duplicated values
df.duplicated().any()

# Missing values in the 'body' column are filled with an empty string
df['body'].fillna('', inplace=True)

# # Check for any missing value again
df.isna().sum()

# A fraction of data is used to the further analysis
subset_fraction = 0.5

# A random 50% sample of the data with reproducibility
df_subset = df.sample(frac=subset_fraction, random_state=42)


# Preprocessing text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove handlers
    text = re.sub(r'@\S+', '', text)

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Remove single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', '', text)

    # Substituting multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)

    # Tokenization
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)


# Apply preprocessing to the 'body' column and assigned it to a new column 'preprocessed_body'
df_subset['preprocessed_body'] = df_subset['body'].apply(preprocess_text)

# Apply preprocessing to the 'title' column and assigned it to a new column 'preprocessed_title'
df_subset['preprocessed_title'] = df_subset['title'].apply(preprocess_text)


# Function for sentiment analysis
def analyze_sentiment(text):
    # initialize NLTK sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    # Get compound sentiment score for the text
    compound_score = sid.polarity_scores(text)['compound']
    # Assign sentiment: positive, negative, or neutral based on the compound score
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'


# Apply sentiment analysis to the 'preprocessed_body' column and assigned it to a new column
df_subset['sentiment_body'] = df_subset['preprocessed_body'].apply(analyze_sentiment)

# Apply sentiment analysis to the 'preprocessed_title' column and assigned it to a new column
df_subset['sentiment_title'] = df_subset['preprocessed_title'].apply(analyze_sentiment)

# Print the value counts for 'sentiment_body'
print(df_subset['sentiment_body'].value_counts())

# Plot bar chart for 'sentiment_body'
plt.figure(figsize=(8, 5))
sns.countplot(x='sentiment_body', data=df_subset, order=df_subset['sentiment_body'].value_counts().index)
plt.title('Sentiment Distribution in Body')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Print the value counts for 'sentiment_title'
print(df_subset['sentiment_title'].value_counts())

# Plot bar chart for 'sentiment_title'
plt.figure(figsize=(8, 5))
sns.countplot(x='sentiment_title', data=df_subset, order=df_subset['sentiment_title'].value_counts().index)
plt.title('Sentiment Distribution in Title')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Count sentiments for 'body'
sentiment_body_counts = df_subset['sentiment_body'].value_counts()

# Count sentiments for 'title'
sentiment_title_counts = df_subset['sentiment_title'].value_counts()

# Create a DataFrame for sentiment counts
sentiment_counts_df = pd.DataFrame({
    'Sentiment Body': sentiment_body_counts,
    'Sentiment Title': sentiment_title_counts
})

# Count empty strings in 'body' column
empty_body_count = (df_subset['body'] == '').sum()
# Print the result
print(f"Number of cells with an empty string in the 'body' column: {empty_body_count}")

# Count empty strings in 'title' column
empty_title_count = (df_subset['title'] == '').sum()
# Print the result
print(f"Number of cells with an empty string in the 'title' column: {empty_title_count}")

# Create a list of tokenized documents for the subset
tokenized_documents_subset = df_subset['preprocessed_title'].apply(word_tokenize)

# Create a dictionary representation of the documents
dictionary_subset = corpora.Dictionary(tokenized_documents_subset)

# The document-term matrix
doc_term_matrix_subset = [dictionary_subset.doc2bow(doc) for doc in tokenized_documents_subset]

# Build the LDA model for the subset
lda_model_subset = gensim.models.LdaModel(
    corpus=doc_term_matrix_subset,
    id2word=dictionary_subset,
    num_topics=10,
    random_state=100,
    passes=10,
    per_word_topics=True
)

# Print the topics for the subset
pprint(lda_model_subset.print_topics())

# Calculate coherence score for the subset
coherence_model_lda_subset = CoherenceModel(model=lda_model_subset, texts=tokenized_documents_subset,
                                            dictionary=dictionary_subset, coherence='c_v')

coherence_lda_subset = coherence_model_lda_subset.get_coherence()

print('Coherence Score for Subset:', coherence_lda_subset)


# Used guide from
# https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0
# to evaluate the LDA model

# Creating a support function to compute the coherence
# k is number of topics, alpha and beta are hyperparameters
def compute_coherence_values(corpus, dictionary, k, a, b):
    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=k,
        random_state=100,
        passes=10,
        alpha=a,
        eta=b,
        per_word_topics=True
    )

    coherence_model_lda = CoherenceModel(model=lda_model, texts=tokenized_documents_subset,
                                         dictionary=dictionary_subset, coherence='c_v')

    return coherence_model_lda.get_coherence()


# Initialize the grid dictionary and add a key to it
grid = {'Validation_Set': {}}

# Specify topics range
min_topics = 2
max_topics = 11
step_size = 1
topics_range = range(min_topics, max_topics, step_size)

# Specify range of alpha parameter
alpha = list(np.arange(0.01, 1, 0.3))
alpha.append('symmetric')
alpha.append('asymmetric')

# Specify range of beta parameter
beta = list(np.arange(0.01, 1, 0.3))
beta.append('symmetric')

# Preparing validations set (one with 75% and the other with 100%)
num_of_docs = len(doc_term_matrix_subset)
corpus_sets = [gensim.utils.ClippedCorpus(doc_term_matrix_subset, int(num_of_docs * 0.75)),
               doc_term_matrix_subset]

corpus_title = ['75% Corpus', '100% Corpus']

# Creating empty dictionaries for results for hyperparameter tuning
model_results = {'Validation_Set': [],
                 'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': []
                 }

# Perform grid search
if 1 == 1:
    pbar = tqdm.tqdm(total=(len(beta) * len(alpha) * len(topics_range) * len(corpus_title)))

    # iterate through validation corpus
    for i in range(len(corpus_sets)):
        # iterate through number of topics
        for k in topics_range:
            # iterate through alpha values
            for a in alpha:
                # iterare through beta values
                for b in beta:
                    # get the coherence score for the given parameters
                    cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=dictionary_subset,
                                                  k=k, a=a, b=b)
                    # Save the model results
                    model_results['Validation_Set'].append(corpus_title[i])
                    model_results['Topics'].append(k)
                    model_results['Alpha'].append(a)
                    model_results['Beta'].append(b)
                    model_results['Coherence'].append(cv)

                    pbar.update(1)
    # save result to CSV
    pd.DataFrame(model_results).to_csv('C:/Users/Handan/Desktop/lda_tuning_results.csv', index=False)
    pbar.close()


# Read result CSV file into a data frame
df_result = pd.read_csv('lda_tuning_results.csv')
df_result.info()

# Find the row with the highest coherence score
max_coherence_row = df_result.loc[df_result['Coherence'].idxmax()]

# Print the hyperparameters for the optimal result
print("Optimal Hyperparameters:")
print(f"Validation Set: {max_coherence_row['Validation_Set']}")
print(f"Number of Topics: {max_coherence_row['Topics']}")
print(f"Alpha: {max_coherence_row['Alpha']}")
print(f"Beta: {max_coherence_row['Beta']}")
print(f"Coherence Score: {max_coherence_row['Coherence']}")

# Build the LDA model with optimal hyperparameters
optimal_lda_model = gensim.models.LdaModel(
    corpus=doc_term_matrix_subset,
    id2word=dictionary_subset,
    num_topics=10,
    random_state=100,
    passes=10,
    alpha='asymmetric',
    eta=0.61,
    per_word_topics=True
)

# Visualization for the subset
vis_subset = pyLDAvis.gensim.prepare(lda_model_subset, doc_term_matrix_subset, dictionary_subset)

# Visualization for the subset in an HTML file
pyLDAvis.save_html(vis_subset, 'lda_visualization_subset.html')
