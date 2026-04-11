import nltk
from nltk.corpus import stopwords

# Ensure the stopwords dataset is downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

STOP_WORDS = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """
    Filters words case-insensitively against the NLTK English stopwords set.
    """
    if not text:
        return ""
    return " ".join([word for word in text.split() if word.lower() not in STOP_WORDS])
