from flask import Flask, render_template, request
import pickle
import nltk
import re
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

# ---------- Download ONLY required NLTK data ----------
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

# ---------- Load stopwords & stemmer ----------
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# ---------- Load trained model & vectorizer ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

tfidf = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))
model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))

# ---------- Text Preprocessing (SAFE TOKENIZER) ----------
def transform_text(text):
    text = text.lower()

    # Regex tokenizer (NO punkt / punkt_tab needed)
    tokens = re.findall(r'\b\w+\b', text)

    cleaned = []
    for word in tokens:
        if word not in stop_words:
            cleaned.append(ps.stem(word))

    return " ".join(cleaned)

# ---------- Routes ----------
@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        message = request.form.get("message", "").strip()

        if not message:
            result = "Please enter a message ❗"
        else:
            transformed_sms = transform_text(message)
            vector_input = tfidf.transform([transformed_sms])
            prediction = model.predict(vector_input)[0]
            result = "Spam 🚫" if prediction == 1 else "Not Spam ✅"

    return render_template("index.html", result=result)

# ---------- Entry Point ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
