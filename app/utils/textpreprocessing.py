import spacy
import html 
import re 
from unidecode import unidecode
from tqdm import tqdm 

class TextPreprocessor:
    """
    A class for text preprocessing.
    """
    def __init__(self):
        """
        Initializes the TextPreprocessor object.
        """
        try:
            self.nlp = spacy.load("model/spacy") # use spacy cache
        except:
            self.nlp = spacy.load("en_core_web_sm")
        tqdm.pandas()

    def lemmatize_text(self, text: str) -> list:
        """
        Lemmatizes the input text.

        Args:
        - text (str): The input text to lemmatize.

        Returns:
        - list: A list of lemmatized tokens.
        """
        doc = self.nlp(text)
        return [token.lemma_ for token in doc]

    def remove_stopwords(self, tokens: list) -> list:
        """
        Removes stopwords from the input list of tokens.

        Args:
        - tokens (list): The list of tokens.

        Returns:
        - list: The list of tokens with stopwords removed.
        """
        return [token for token in tokens if not self.nlp.vocab[token].is_stop]

    def remove_punctuations(self, tokens: list) -> list:
        """
        Removes punctuations from the input list of tokens.

        Args:
        - tokens (list): The list of tokens.

        Returns:
        - list: The list of tokens with punctuations removed.
        """
        return [token for token in tokens if not self.nlp.vocab[token].is_punct]

    def decode_html_entities(self, text: str) -> str:
        """
        Decodes HTML entities in the input text.

        Args:
        - text (str): The input text.

        Returns:
        - str: The text with HTML entities decoded.
        """
        return html.unescape(text)

    def remove_html_tags(self, text: str) -> str:
        """
        Removes HTML tags from the input text.

        Args:
        - text (str): The input text.

        Returns:
        - str: The text with HTML tags removed.
        """
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', text)
        return cleantext

    def preprocess_text(self, text: str) -> list:
        """
        Preprocesses the input text.

        Args:
        - text (str): The input text to preprocess.

        Returns:
        - list: The preprocessed tokens.
        """
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
