from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import joblib
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Load pre-trained logistic regression model
model = joblib.load("fake_news_detection_model.pkl")

# Preprocessing setup
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase and remove punctuation
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    # Tokenize, remove stopwords, and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

def split_text_sliding_window(text, chunk_size=100, overlap=20):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.split()) >= chunk_size * 0.5:  # Ensure chunk has sufficient length
            chunks.append(chunk)
    return chunks

def classify_long_text(text, chunk_size=100, overlap=20):
    chunks = split_text_sliding_window(text, chunk_size=chunk_size, overlap=overlap)
    chunk_predictions = []

    for chunk in chunks:
        preprocessed_chunk = preprocess_text(chunk)
        prediction = model.predict([preprocessed_chunk])[0]
        chunk_predictions.append(prediction)
    if 1 in chunk_predictions:
        overall_result = "Fakes News Not Detected"
    else:
        overall_result = "Fake News Detected"
    
    return overall_result

def fetch_article_content(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            text = ' '.join([para.get_text() for para in paragraphs])
            return text
        else:
            print(f"Failed to fetch {url} - Status Code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    content = request.get_json()
    url = content.get("url")
    
    if not url:
        return jsonify({"error": "URL not provided"}), 400

    # Fetch article content
    article_text = fetch_article_content(url)
    if not article_text:
        return jsonify({"error": "Unable to fetch content from URL"}), 500

    # Classify the content
    result = classify_long_text(article_text)
    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(port = 5000, debug=True)
