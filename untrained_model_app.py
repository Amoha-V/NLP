import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Download necessary NLTK resources
nltk.download('stopwords', quiet=True)

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
        try:
            df = pd.read_csv(file_path, encoding='latin-1')
            df = df.dropna(how='any', axis=1)
            
            # Check and rename columns if necessary
            if len(df.columns) == 2:
                df.columns = ['target', 'message']
            elif 'label' in df.columns:
                df = df.rename(columns={'label': 'target'})
            
            return df
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            return None

    def prepare_data(self, df):
        """Prepare data for training"""
        df['message_clean'] = df['message'].apply(self.preprocess_text)
        
        X = df['message_clean']
        y = (df['target'].str.lower() == 'spam').astype(int)
        
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, X_train, y_train):
        """Train Naive Bayes pipeline"""
        self.model = Pipeline([
            ('bow', CountVectorizer()),
            ('tfid', TfidfTransformer()),
            ('classifier', MultinomialNB())
        ])
        
        self.model.fit(X_train, y_train)

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
    st.title('SMS Spam Detector 🕵️‍♀️')
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    # Initialize the detector
    detector = SMSSpamDetector()
    
    # Training status flag
    model_trained = False
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp_spam.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load data
        df = detector.load_data("temp_spam.csv")
        
        if df is not None:
            # Show dataset preview
            st.subheader("Dataset Preview")
            st.dataframe(df.head())
            
            # Split and train
            X_train, X_test, y_train, y_test = detector.prepare_data(df)
            detector.train_model(X_train, y_train)
            
            # Model metrics
            st.subheader("Model Performance")
            y_pred = detector.model.predict(X_test)
            
            from sklearn.metrics import accuracy_score, classification_report
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {accuracy:.2%}")
            
            # Detailed classification report
            st.text("Classification Report:")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            
            model_trained = True
            
            # Remove temporary file
            os.remove("temp_spam.csv")
    
    # Spam detection section
    st.subheader('Spam Detection')
    user_input = st.text_area('Enter your message:', height=200)
    
    if st.button('Check Spam'):
        if not model_trained:
            st.warning('Please upload a training dataset first!')
        elif user_input:
            # Make prediction
            result = detector.predict(user_input)
            
            # Display results with styling
            if result['is_spam']:
                st.error('🚨 This message looks like SPAM!')
                st.warning(f'Spam Probability: {result["spam_probability"]:.2%}')
            else:
                st.success('✅ This message appears to be safe (not spam).')
                st.info(f'Spam Probability: {result["spam_probability"]:.2%}')
        else:
            st.warning('Please enter a message to check.')
    
    # Sidebar information
    st.sidebar.header('About')
    st.sidebar.info(
        'This Spam Detector uses Natural Language Processing (NLP) '
        'and Machine Learning to identify spam messages. '
        'Upload your own dataset to train the model!'
    )

if __name__ == '__main__':
    main()