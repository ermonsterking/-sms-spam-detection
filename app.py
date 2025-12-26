from flask import Flask, render_template, request
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load stopwords once
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

app = Flask(__name__)

# Load trained model & vectorizer
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# ---------- Text Preprocessing ----------
def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    cleaned = []
    for word in tokens:
        if word.isalnum() and word not in stop_words:
            cleaned.append(ps.stem(word))

    return " ".join(cleaned)

# ---------- Routes ----------
@app.route("/", methods=["GET", "POST"])
def home():
    result = ""

    if request.method == "POST":
        message = request.form["message"]

        transformed_sms = transform_text(message)
        vector_input = tfidf.transform([transformed_sms])
        prediction = model.predict(vector_input)[0]

        result = "Spam 🚫" if prediction == 1 else "Not Spam ✅"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
