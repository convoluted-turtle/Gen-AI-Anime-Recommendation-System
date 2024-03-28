import spacy
import html 
import re 
from unidecode import unidecode
from tqdm import tqdm 
import sys

print(sys.path)

class TextPreprocessor:
    def __init__(self):
        self.nlp = spacy.load("model/spacy")
        tqdm.pandas()

    def lemmatize_text(self, text):
        doc = self.nlp(text)
        return [token.lemma_ for token in doc]

    def remove_stopwords(self, tokens):
        return [token for token in tokens if not self.nlp.vocab[token].is_stop]

    def remove_punctuations(self, tokens):
        return [token for token in tokens if not self.nlp.vocab[token].is_punct]

    def decode_html_entities(self, text):
        return html.unescape(text)

    def remove_html_tags(self, text):
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', text)
        return cleantext

    def preprocess_text(self, text):
        # Lowercase text
        text = text.lower()

        # Decode HTML entities
        text = self.decode_html_entities(text)

        # Remove HTML tags
        text = self.remove_html_tags(text)

        # Normalize Unicode characters
        text = unidecode(text)

        # Lemmatize text
        tokens = self.lemmatize_text(text)

        # Remove stopwords
        tokens = self.remove_stopwords(tokens)

        # Remove punctuations
        tokens = self.remove_punctuations(tokens)

        return tokens