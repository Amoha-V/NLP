import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download necessary NLTK resources
nltk.download('stopwords')

class SMSSpamDetector:
    def __init__(self):
        self.stop_words = set(stopwords.words('english') + ['u', 'im', 'c'])
        self.stemmer = SnowballStemmer("english")
        self.model = None

    def clean_text(self, text):
        """Clean and preprocess text"""
        text = str(text).lower()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
        text = re.sub(r'\n', '', text)
        text = re.sub(r'\w*\d\w*', '', text)
        return text

    def remove_stopwords(self, text):
        """Remove stopwords from text"""
        return ' '.join(word for word in text.split() if word not in self.stop_words)

    def stem_text(self, text):
        """Apply stemming to text"""
        return ' '.join(self.stemmer.stem(word) for word in text.split())

    def preprocess_text(self, text):
        """Full preprocessing pipeline"""
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        text = self.stem_text(text)
        return text

    def load_data(self, file_path):
        """Load spam dataset"""
        df = pd.read_csv(file_path, encoding='latin-1')
        df = df.dropna(how='any', axis=1)
        df.columns = ['target', 'message']
        return df

    def prepare_data(self, df):
        """Prepare data for training"""
        df['message_clean'] = df['message'].apply(self.preprocess_text)
        
        X = df['message_clean']
        y = (df['target'] == 'spam').astype(int)
        
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, X_train, y_train):
        """Train Naive Bayes pipeline"""
        self.model = Pipeline([
            ('bow', CountVectorizer()),
            ('tfid', TfidfTransformer()),
            ('classifier', MultinomialNB())
        ])
        
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)
        
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    def predict(self, message):
        """Predict spam for a single message"""
        if self.model is None:
            raise ValueError("Model must be trained first!")
        
        processed_message = self.preprocess_text(message)
        prediction = self.model.predict([processed_message])[0]
        probability = self.model.predict_proba([processed_message])[0][1]
        
        return {
            'is_spam': bool(prediction),
            'spam_probability': probability
        }

def main():
    # Example usage
    detector = SMSSpamDetector()
    
    # Replace with your actual dataset path
    df = detector.load_data('spam.csv')
    
    X_train, X_test, y_train, y_test = detector.prepare_data(df)
    
    detector.train_model(X_train, y_train)
    detector.evaluate_model(X_test, y_test)
    
    # Example predictions
    test_messages = [
        "Congratulations! You've won a free iPhone!",
        "Hey, can we meet for coffee later?"
    ]
    
    for message in test_messages:
        result = detector.predict(message)
        print(f"\nMessage: {message}")
        print(f"Spam: {result['is_spam']}")
        print(f"Spam Probability: {result['spam_probability']:.2%}")

if __name__ == "__main__":
    main()