import re
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



class TextPreprocessorCNN:
    def __init__(
        self,
        num_words=20000,
        max_len=300,
    ):
        """
        Parameters:
        - num_words: Max vocabulary size for tokenizer
        - max_len: Max padded sequence length
        - remove_stopwords: Whether to remove stopwords
        """
        self.num_words = num_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=self.num_words)


    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)  
        text = re.sub(r"\s+", " ", text).strip()  
        return text

    def fit(self, df):
        texts = df.iloc[:, 0].astype(str).apply(self.clean_text).tolist()
        self.tokenizer.fit_on_texts(texts)

    def transform(self, df):
        texts = df.iloc[:, 0].astype(str).apply(self.clean_text).tolist()
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len, padding="post")
        return padded

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
