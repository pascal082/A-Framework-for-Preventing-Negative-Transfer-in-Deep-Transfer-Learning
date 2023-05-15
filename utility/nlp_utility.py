import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string



# Set up NLTK resources
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

# Set up stop words and punctuation removal
stop_words = set(stopwords.words("english"))
punctuations = set(string.punctuation)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


# Preprocess the text dataframe
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)

    # Remove stop words and punctuation
    tokens = [token.lower() for token in tokens if token.lower() not in stop_words and token not in punctuations]

    # Lemmatization
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]

    return lemmas


