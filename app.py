from flask import Flask, render_template, request
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---------------- NLTK SAFE DOWNLOAD (RENDER FIX) ----------------
def download_nltk_resources():
    resources = [
        ('corpora/stopwords', 'stopwords'),
        ('tokenizers/punkt', 'punkt')
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)

download_nltk_resources()
# ----------------------------------------------------------------

app = Flask(__name__)

# Load stopwords & stemmer
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

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
    result = None

    if request.method == "POST":
        message = request.form.get("message", "")

        transformed_sms = transform_text(message)
        vector_input = tfidf.transform([transformed_sms])
        prediction = model.predict(vector_input)[0]

        result = "Spam 🚫" if prediction == 1 else "Not Spam ✅"

    return render_template("index.html", result=result)

# ---------- Entry Point ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
