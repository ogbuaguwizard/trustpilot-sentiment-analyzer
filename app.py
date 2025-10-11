import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from textblob import TextBlob
import nltk
import time
import re

# Ensure necessary NLTK data
def ensure_nltk_data():
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        with st.spinner("🔄 Setting up NLP data (first run only)..."):
            nltk.download('punkt')
            nltk.download('punkt_tab')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('averaged_perceptron_tagger_eng')
            nltk.download('brown')
            nltk.download('wordnet')
            nltk.download('movie_reviews')

ensure_nltk_data()

# ======================
# Sentiment & Aspect Functions
# ======================
def get_sentiment(text):
    blob = TextBlob(text)
    score = blob.sentiment.polarity
    if score > 0.1:
        return "positive", score
    elif score < -0.1:
        return "negative", score
    else:
        return "neutral", score

def extract_aspects(text):
    blob = TextBlob(text)
    nouns = [word.lower() for word, tag in blob.tags if tag.startswith("NN")]
    return nouns

# ======================
# Scraper
# ======================
def scrape_trustpilot(domain):
    base_url = f"https://www.trustpilot.com/review/{domain.strip()}"
    page = 1
    all_reviews = []

    progress = st.progress(0)
    while True:
        url = f"{base_url}?page={page}"
        st.write(f"🔍 Scraping page {page}: {url}")
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})

        if response.status_code != 200:
            st.warning(f"⚠️ Failed to fetch page {page}, stopping.")
            break

        soup = BeautifulSoup(response.text, "html.parser")
        sections = soup.find_all("section", class_="styles_reviewContentwrapper__K2aRu")

        if not sections:
            st.success("✅ No more reviews found. Stopping.")
            break

        for section in sections:
            rating_tag = section.find("div", class_="styles_reviewHeader__DzoAZ")
            rating = rating_tag.get("data-service-review-rating") if rating_tag else None

            title_tag = section.find("h2")
            title = title_tag.get_text(strip=True) if title_tag else None

            review_tag = section.find("p")
            review = review_tag.get_text(strip=True) if review_tag else None

            date_tag = section.find("div", {"data-testid": "review-badge-date"})
            date = date_tag.get_text(strip=True) if date_tag else None

            if review:
                all_reviews.append({
                    "rating": rating,
                    "title": title,
                    "review": review,
                    "date": date
                })

        progress.progress(min(page / 10, 1.0))
        st.write(f"✅ Page {page} scraped ({len(sections)} reviews).")
        page += 1
        time.sleep(1.5)

    return pd.DataFrame(all_reviews)

# ======================
# STREAMLIT UI
# ======================
st.set_page_config(page_title="Trustpilot Sentiment Analyzer", page_icon="💬", layout="wide")

st.title("💬 Trustpilot Review Sentiment Analyzer")
st.write("Enter a domain (e.g. `facebook.com`, `www.amazon.com`) to scrape and analyze recent Trustpilot reviews in real time.")

# Input box
domain = st.text_input("🌐 Enter website domain:", "www.facebook.com")

if st.button("🚀 Analyze"):
    with st.spinner(f"Scraping reviews for **{domain}**..."):
        df_raw = scrape_trustpilot(domain)

    if df_raw.empty:
        st.error("⚠️ No reviews found — please check the URL or structure.")
    else:
        st.success(f"✅ Scraped {len(df_raw)} reviews successfully!")

        st.subheader("📄 Raw Scraped Data")
        st.dataframe(df_raw)

        with st.spinner("🔍 Performing sentiment and aspect analysis..."):
            df_analyzed = df_raw.copy()
            df_analyzed[["sentiment", "score"]] = df_analyzed["review"].apply(
                lambda text: pd.Series(get_sentiment(text))
            )
            df_analyzed["aspects"] = df_analyzed["review"].apply(extract_aspects)

        st.subheader("📊 After Sentiment Analysis")
        st.dataframe(df_analyzed)

        # Sentiment distribution
        st.subheader("📈 Sentiment Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        df_analyzed["sentiment"].value_counts().plot(kind="bar", ax=ax, color=["green", "red", "gray"])
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Sentiments")
        st.pyplot(fig)

        # Aspect frequency
        all_nouns = sum(df_analyzed["aspects"], [])
        common_aspects = Counter(all_nouns).most_common(10)
        absa_df = pd.DataFrame(common_aspects, columns=["Aspect", "Frequency"])

        st.subheader("🔍 Top 10 Most Frequent Aspects")
        st.dataframe(absa_df)

        # Final summary
        pos = (df_analyzed["sentiment"] == "positive").sum()
        neg = (df_analyzed["sentiment"] == "negative").sum()
        neu = (df_analyzed["sentiment"] == "neutral").sum()
        total = pos + neg + neu
        final_score = (pos - neg) / total if total > 0 else 0
        overall = "positive" if final_score > 0.05 else "negative" if final_score < -0.05 else "neutral"

        st.markdown(f"""
        ### 🧾 Final Summary
        - 👍 Positive Reviews: **{pos}**
        - 👎 Negative Reviews: **{neg}**
        - 😐 Neutral Reviews: **{neu}**
        - 🌟 **Overall Sentiment:** `{overall.upper()}`  
        - 📊 **Sentiment Score:** `{final_score:.2f}`
        """)

