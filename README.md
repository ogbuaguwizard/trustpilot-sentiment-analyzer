# 🌐 Trustpilot Sentiment Analyzer  
### Real-Time Opinion Mining and Aspect-Based Sentiment Analysis (ABSA) App  

🔗 **Live App:** [https://website-sentiment-analyzer.streamlit.app/](https://website-sentiment-analyzer.streamlit.app/)  
💻 **GitHub Repository:** [https://github.com/ogbuaguwizard/trustpilot-sentiment-analyzer/](https://github.com/ogbuaguwizard/trustpilot-sentiment-analyzer/)

---

## 📖 Overview  

The **Trustpilot Sentiment Analyzer** is a **Streamlit-based ABSA (Aspect-Based Sentiment Analysis)** web application that scrapes and analyzes **Trustpilot reviews** for any website (e.g., `facebook.com`, `airbnb.com`).  

It performs **real-time opinion mining** and extracts **key aspects** (topics) with their associated sentiments, helping to evaluate websites based on genuine user feedback.  

This project is part of a broader study focused on **designing a comprehensive framework for website evaluation using opinion mining techniques**, ensuring **accuracy**, **real-time adaptability**, and **scalable insights**.

---

## 🚀 Features  

✅ **Real-time scraping** — Automatically fetches multiple Trustpilot review pages until no more reviews are available.  
✅ **Aspect-based opinion mining (ABSA)** — Identifies key aspects (nouns) and associated opinions (adjectives). 
✅ **Sentiment analysis** — Classifies each pair into *positive*, *neutral*, or *negative* using TextBlob.
✅ **Interactive Streamlit UI** — Simple input for website domain, real-time results, and dynamic controls.  
✅ **Color-coded sentiment boxes** — Easy visualization of extracted opinions.  
✅ **Visual insights** — Pie chart of sentiment distribution with clean, white-labeled charts.  
✅ **Overall sentiment summary** — Combines analysis results into an intuitive rating summary.

---

## 🧠 Core Technologies  

| Category | Libraries/Tools |
|-----------|----------------|
| **Frontend (UI)** | Streamlit |
| **Web Scraping** | BeautifulSoup4, Requests |
| **Data Processing** | Pandas |
| **Sentiment Analysis** | TextBlob, NLTK |
| **Visualization** | Matplotlib |
| **Aspect Mining (ABSA)** | POS tagging via NLTK and TextBlob |

---

## ⚙️ Installation  

To set up the app locally:  

```bash
# Clone the repository
git clone https://github.com/ogbuaguwizard/trustpilot-sentiment-analyzer.git
cd trustpilot-sentiment-analyzer

# Install dependencies
pip install -r requirements.txt
```

Running the App
```bash
streamlit run app.py
```

Citation (if used in research)
```bash
Ogbuagu, F. K. (2025). Design and Implementation of a Comprehensive Framework for Website Evaluation Using Opinion Mining Techniques.
```
