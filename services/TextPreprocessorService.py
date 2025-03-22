import kagglehub
from kagglehub import KaggleDatasetAdapter
import re
from constants import Constants
import nltk
from nltk.corpus import stopwords




class TextPreprocessorService:

    def __init__(self):
        self.load_text_data()
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))

    
    def load_text_data(self):
        self.text_dataset = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            Constants.DATASET_NAME,
            Constants.TEXT_DATA_FILENAME
        )


    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        tokens = [word for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)
    
    def produce(self):
        self.text_dataset['cleaned_text'] = self.text_dataset['Caption'].apply(self.preprocess_text)
        self.text_dataset['LABEL'] = self.text_dataset['LABEL'].map({'neutral': 0, 'positive': 1, 'negative': 2})
        return self.text_dataset    