# 🌐 Trustpilot Sentiment Analyzer  
### Real-Time Opinion Mining and Aspect-Based Sentiment Analysis (ABSA) App

---

## 📖 Overview

This project is a **Streamlit-based web application** that scrapes **Trustpilot reviews** for any website (e.g., `facebook.com`, `airbnb.com`) and performs **real-time sentiment analysis** and **aspect-based review summarization (ABSA)**.

It was developed as part of a study aimed at designing a comprehensive framework for **website evaluation using opinion mining techniques** for accurate, real-time, and scalable feedback analysis.

---

## 🚀 Features

✅ **Real-time scraping** — fetches multiple Trustpilot pages until no more reviews are found.  
✅ **Sentiment Analysis** — uses `TextBlob` to classify each review as *positive*, *negative*, or *neutral*.  
✅ **Aspect Extraction (ABSA)** — identifies frequent nouns (topics) discussed in reviews.  
✅ **Visual Charts** — displays sentiment distribution using Matplotlib.  
✅ **Final Summary** — produces an overall website sentiment rating.  
✅ **Streamlit UI** — users can simply enter a website domain and view analysis instantly.

---

## 🧠 Core Technologies

| Category | Libraries/Tools |
|-----------|----------------|
| **Frontend (UI)** | Streamlit |
| **Web Scraping** | BeautifulSoup4, Requests |
| **Data Processing** | Pandas |
| **Sentiment Analysis** | TextBlob, NLTK |
| **Visualization** | Matplotlib |
| **ABSA (Aspect Mining)** | TextBlob POS tagging |

---

## ⚙️ Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/<your-username>/trustpilot-sentiment-analyzer.git
cd trustpilot-sentiment-analyzer
pip install -r requirements.txt
```
Running the App
```bash
streamlit run app.py
```

Project Structure
trustpilot-sentiment-analyzer/
│
├── app.py
├── requirements.txt
└── README.md
