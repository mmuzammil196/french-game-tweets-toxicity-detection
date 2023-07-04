from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

tfidf = TfidfVectorizer(max_features=3000)
vectorizer = pickle.load(open('vectorizer', 'rb'))
model = pickle.load(open('game_tweet_model.pkl', 'rb'))

nltk.download('stopwords')
nltk.download('punkt')

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()  # 1. Convert to lowercase
    text = nltk.word_tokenize(text)  # 2. Tokenize

    y = []
    for i in text:
        if i.isalnum():  # 3. Remove special characters
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:  # 4. Remove stopwords and punctuation
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))  # 5. Stemming

    return " ".join(y)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text')
    tweet = transform_text(text)
    result = model.predict(vectorizer.transform([tweet]))
    toxicity_label = "toxic" if result[0] == 1 else "not toxic"
    return render_template('index.html', result=toxicity_label)


if __name__ == '__main__':
    app.run(debug=True)
