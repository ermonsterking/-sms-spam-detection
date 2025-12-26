

---

# 📩 SMS Spam Detection Web App

A **Machine Learning–based SMS Spam Detection system** deployed using **Flask**, which classifies messages as **Spam 🚫** or **Not Spam ✅** in real time using **Natural Language Processing (NLP)** techniques.

---

## 🚀 Project Overview

Spam messages often contain promotional, fraudulent, or misleading content.
This project uses **TF-IDF vectorization** and a **trained ML classifier** to automatically detect spam SMS messages through a simple and user-friendly web interface.

---

## 🧠 Machine Learning Pipeline

1. **Text Preprocessing**

   * Lowercasing
   * Tokenization
   * Stopword removal
   * Stemming (Porter Stemmer)

2. **Feature Extraction**

   * TF-IDF Vectorizer (`max_features=3000`)

3. **Model Training**

   * Multinomial Naive Bayes
     *(Can be extended to Voting / Ensemble models)*

4. **Evaluation Metrics**

   * Accuracy
   * Precision (priority metric for spam detection)

5. **Deployment**

   * Flask Web Application
   * Model & vectorizer loaded using `pickle`

---

## 🖥️ Web Application Features

* Clean & responsive UI
* Real-time spam prediction
* Emoji-based result display
* Result shown **only after submission**
* Lightweight and fast inference

---

## 🗂️ Project Structure

```
sms-spam-detection/
│
├── app.py                  # Flask backend
├── model.pkl               # Trained ML model
├── vectorizer.pkl          # TF-IDF vectorizer
├── requirements.txt        # Project dependencies
│
├── templates/
│   └── index.html          # Frontend UI
│
└── static/
    └── style.css           # Styling
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/sms-spam-detection.git
cd sms-spam-detection
```

---

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Download NLTK Resources (Run Once)

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

### 5️⃣ Run the Application

```bash
python app.py
```

Open browser:

```
http://127.0.0.1:5000
```

---

## 🧪 Example Test Messages

### Spam 🚫

```
Congratulations! You won ₹1,00,000. Claim now!
```

### Not Spam ✅

```
Hey, are we meeting tomorrow at 10?
```

---

## 📊 Why Precision Matters

In spam detection:

* **False Positives (ham → spam)** are costly
* Precision ensures **important messages are never lost**

> This project prioritizes **high precision** to reduce false spam alerts.

---

## 🛠️ Technologies Used

* Python
* Flask
* Scikit-learn
* NLTK
* HTML, CSS
* Pickle (Model persistence)

---

## 📌 Future Improvements

* Add confidence score (% spam)
* Use Voting / Stacking classifiers
* Deploy on Render / AWS / Azure
* Add REST API support
* Mobile-friendly UI

---

## 👨‍💻 Author

**Avineesh Kumar**
CSE Student | Frontend & ML Enthusiast
Bhagalpur College of Engineering

---

## 📜 License

This project is open-source and available for educational purposes.

---

### ⭐ If you like this project, don’t forget to star the repository!

---



