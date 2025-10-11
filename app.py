import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from textblob import TextBlob
import nltk
import time
import re

st.set_page_config(page_title="Trustpilot ABSA Analyzer", page_icon="💬", layout="wide")

# ======================
# Setup NLTK
# ======================
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
# Aspect Extraction & Sentiment
# ======================
def extract_aspects_and_opinions(text):
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)

    triples = []
    for i, (word, tag) in enumerate(tagged):
        if tag.startswith("NN"):  # noun = aspect
            left_adj = tagged[i - 1][0] if i > 0 and tagged[i - 1][1].startswith("JJ") else None
            right_adj = tagged[i + 1][0] if i < len(tagged) - 1 and tagged[i + 1][1].startswith("JJ") else None
            opinion = left_adj or right_adj
            if opinion:
                blob = TextBlob(opinion)
                sentiment_score = blob.sentiment.polarity
                sentiment_label = (
                    "positive" if sentiment_score > 0.1 else
                    "negative" if sentiment_score < -0.1 else
                    "neutral"
                )
                triples.append((word.lower(), opinion.lower(), sentiment_label))
    return triples


def analyze_aspects(df):
    aspect_sentiments = defaultdict(list)
    aspect_opinion_per_review = []

    for _, row in df.iterrows():
        review = row["review"]
        pairs = extract_aspects_and_opinions(review)
        aspect_opinion_per_review.append(pairs)
        for aspect, opinion, sentiment in pairs:
            aspect_sentiments[aspect].append(sentiment)

    # Summarize per aspect
    aspect_summary = []
    for aspect, sentiments in aspect_sentiments.items():
        pos = sentiments.count("positive")
        neg = sentiments.count("negative")
        neu = sentiments.count("neutral")
        total = pos + neg + neu
        dominant = max(["positive", "neutral", "negative"], key=lambda s: sentiments.count(s))
        aspect_summary.append({
            "Aspect": aspect,
            "Positive": pos,
            "Neutral": neu,
            "Negative": neg,
            "Total": total,
            "Dominant Sentiment": dominant
        })

    return pd.DataFrame(aspect_summary).sort_values("Total", ascending=False), aspect_opinion_per_review


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
st.title("💬 Trustpilot Aspect-Based Sentiment Analyzer (ABSA)")
st.write("Enter a domain (e.g. `facebook.com`, `www.amazon.com`) to scrape and analyze reviews using Aspect-Based Sentiment Analysis (ABSA).")

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

        with st.spinner("🔍 Extracting aspects and analyzing opinions..."):
            aspect_df, per_review_aspects = analyze_aspects(df_raw)
            df_raw["aspect_opinions"] = per_review_aspects

        # ======================
        # Aspect Summary
        # ======================
        st.subheader("🔍 Aspect-Based Sentiment Summary")
        st.dataframe(aspect_df)

        # ======================
        # Aspect/Opinion per Review (Color Coded)
        # ======================
        st.subheader("💡 Aspects & Opinions per Review")
        for i, row in df_raw.iterrows():
            st.markdown(f"### 🗒️ Review {i+1}: {row['title'] or '(no title)'}")
            st.write(row["review"])
            if not row["aspect_opinions"]:
                st.info("No aspect-opinion pairs found.")
                continue

            html_pairs = ""
            for aspect, opinion, sentiment in row["aspect_opinions"]:
                color = {"positive": "#4CAF50", "negative": "#E74C3C", "neutral": "#95A5A6"}[sentiment]
                html_pairs += f"<span style='background:{color}; color:white; padding:3px 8px; border-radius:8px; margin:3px; display:inline-block;'>{aspect} → {opinion} ({sentiment})</span> "
            st.markdown(html_pairs, unsafe_allow_html=True)
            st.divider()

        # ======================
        # Sentiment Distribution (Pie Chart)
        # ======================
        st.subheader("📊 Sentiment Distribution")
        fig, ax = plt.subplots(figsize=(5, 5))
        sizes = [
            aspect_df["Positive"].sum(),
            aspect_df["Neutral"].sum(),
            aspect_df["Negative"].sum()
        ]
        labels = ["Positive", "Neutral", "Negative"]
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'white'})
        ax.set_title("Overall Sentiment Distribution")
        st.pyplot(fig)

        # ======================
        # Overall Sentiment (Dramatic)
        # ======================
        pos_total, neg_total, neu_total = sizes
        total = pos_total + neg_total + neu_total
        final_score = (pos_total - neg_total) / total if total else 0
        overall = "positive" if final_score > 0.05 else "negative" if final_score < -0.05 else "neutral"
        color = {"positive": "#4CAF50", "negative": "#E74C3C", "neutral": "#95A5A6"}[overall]
        emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}[overall]

        st.markdown(f"""
        <div style="text-align:center; padding:20px; border-radius:15px; background-color:{color}; color:white; font-size:28px;">
            <b>🌟 OVERALL SENTIMENT: {overall.upper()} {emoji}</b><br>
            <span style="font-size:18px;">(Score: {final_score:.2f})</span>
        </div>
        """, unsafe_allow_html=True)
