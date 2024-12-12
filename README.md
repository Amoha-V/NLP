# SMS Spam Detection Project

## Project Overview

This SMS Spam Detection application uses Natural Language Processing (NLP) and Machine Learning to classify SMS messages as spam or ham (not spam). The project demonstrates two deployment approaches: an untrained model where users can upload their own dataset and a pre-trained model with a ready-to-use spam detection interface.

## Project Versions

### 1. Untrained Model Version
- **Deployment URL**: [Untrained Model Deployment](https://untrained-model-spam-detection.onrender.com/)
- **Features**:
  - User-driven dataset upload
  - Dynamic model training
  - Interactive performance metrics
  - Flexible spam detection

### 2. Pre-Trained Model Version
- **Deployment URL**: [Trained Model Deployment](https://trained-model-spam-detection.onrender.com/)
- **Features**:
  - Pre-trained on a comprehensive SMS spam dataset
  - Immediate spam detection
  - Low-latency predictions
  - Sidebar with model performance metrics

## Technical Architecture

### Preprocessing Techniques 
- Stopwords removal
- Stemming using SnowballStemmer
- Regular expression-based text sanitization
- Text Normalization
- Tokenization
- Lexical Analysis
- Text Vectorization
- Semantic Processing

## Preprocessing Techniques: A Comprehensive Approach

### Preprocessing Workflow

#### 1. Text Normalization
- **Objectives**:
  - Standardize text representation
  - Remove linguistic noise
  - Prepare text for analysis

- **Key Techniques**:
  - Lowercase conversion
  - Special character removal
  - Numeric character elimination
  - Metadata and markup stripping

```python
def clean_text(text):
    # Lowercase conversion
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespaces
    text = ' '.join(text.split())
    
    return text
```

#### 2. Tokenization
- **Purpose**:
  - Decompose text into meaningful units
  - Prepare for feature extraction
  - Enable granular text analysis

- **Approaches**:
  - Word-level tokenization
  - Character-level tokenization
  - N-gram tokenization

```python
def tokenize_text(text):
    # Word-level tokenization
    tokens = text.split()
    
    # Optional: N-gram generation
    bigrams = list(nltk.bigrams(tokens))
    
    return tokens, bigrams
```

#### 3. Text Vectorization
- **Goal**: 
  - Convert textual data to numerical representations
  - Enable machine learning model processing

- **Techniques**:
  1. **CountVectorizer**
     - Frequency-based representation
     - Creates sparse matrix of token counts

  2. **TF-IDF Vectorization**
     - Weighted term importance
     - Reduces impact of frequently occurring words

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# CountVectorizer
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(texts)

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
```

#### 4. Advanced Text Processing

##### Stopword Elimination
- **Objective**: 
  - Remove common, non-informative words
  - Reduce computational complexity
  - Focus on meaningful content

```python
from nltk.corpus import stopwords

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens
```

##### Stemming
- **Technique**: SnowballStemmer
- **Purpose**:
  - Reduce words to their root form
  - Normalize word variations
  - Compress feature space

```python
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer('english')
def stem_tokens(tokens):
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens
```

### Comprehensive Preprocessing Pipeline

```python
def preprocess_text(text):
    # Normalize
    cleaned_text = clean_text(text)
    
    # Tokenize
    tokens = cleaned_text.split()
    
    # Remove stopwords
    filtered_tokens = remove_stopwords(tokens)
    
    # Stem tokens
    stemmed_tokens = stem_tokens(filtered_tokens)
    
    # Rejoin processed tokens
    processed_text = ' '.join(stemmed_tokens)
    
    return processed_text
```

### Machine Learning Pipeline
- Vectorization: 
  - CountVectorizer
  - TF-IDF Transformation
- Classification Algorithm: Multinomial Naive Bayes

### Key Libraries
- Streamlit (Web Interface)
- Pandas (Data Manipulation)
- Scikit-learn (Machine Learning)
- NLTK (Natural Language Processing)

## Local Setup and Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Installation Steps
1. Clone the repository
```bash
git clone https://github.com/your-username/sms-spam-detector.git
cd sms-spam-detector
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the application
```bash
streamlit run untrained_model_app.py  # For untrained model
# OR
streamlit run trained_model_app.py    # For pre-trained model
```

## Usage Guide

### Untrained Model
1. Upload a CSV file with columns: 'message' and 'target'
2. Model trains automatically on your uploaded dataset
3. View model performance metrics
4. Enter SMS text to check spam probability

### Pre-Trained Model
1. Directly enter SMS text
2. Click "Check Spam"
3. Receive instant spam classification

## Model Performance

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score

## Dataset Recommendations
- Ensure CSV has 'message' and 'target' columns
- 'target' column should contain 'spam' or 'ham' labels
- Recommended dataset: SMS Spam Collection Dataset

## Limitations
- Performance depends on training data quality
- Works best with English language messages
- Limited to SMS-style text inputs

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## License
Distributed under the [MIT License](https://opensource.org/licenses/MIT)
