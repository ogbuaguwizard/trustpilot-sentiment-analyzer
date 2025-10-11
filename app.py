import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from textblob import TextBlob
import nltk
import time
import random

# -------------------------------
# Must be FIRST Streamlit call
# -------------------------------
st.set_page_config(page_title="Trustpilot ABSA Analyzer", page_icon="💬", layout="centered")

# ---------- NLTK setup ----------
def ensure_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        with st.spinner("🔄 Setting up NLP data..."):
            nltk.download('punkt')
            nltk.download('punkt_tab')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('averaged_perceptron_tagger_eng')
            nltk.download('brown')
            nltk.download('wordnet')
            nltk.download('movie_reviews')

ensure_nltk_data()

# ---------- NLP helpers ----------
def extract_aspects_and_opinions(text):
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    pairs = []
    for i, (word, tag) in enumerate(tagged):
        if tag.startswith("NN"):
            left_adj = tagged[i-1][0] if i > 0 and tagged[i-1][1].startswith("JJ") else None
            right_adj = tagged[i+1][0] if i < len(tagged)-1 and tagged[i+1][1].startswith("JJ") else None
            opinion = left_adj or right_adj
            if opinion:
                pairs.append((word.lower(), opinion.lower()))
    return pairs

def get_sentiment_label(score):
    if score > 0.1:
        return "positive"
    elif score < -0.1:
        return "negative"
    else:
        return "neutral"

def analyze_aspects(df):
    aspect_sentiments = defaultdict(list)
    aspect_table = []

    for _, row in df.iterrows():
        pairs = extract_aspects_and_opinions(row["review"])
        pair_list = []
        for aspect, opinion in pairs:
            blob = TextBlob(opinion)
            s = blob.sentiment.polarity
            label = get_sentiment_label(s)
            aspect_sentiments[aspect].append(label)
            pair_list.append((aspect, opinion, label))
        aspect_table.append(pair_list)

    summary = []
    for a, sents in aspect_sentiments.items():
        pos, neu, neg = sents.count("positive"), sents.count("neutral"), sents.count("negative")
        total = pos + neu + neg
        dom = max(["positive", "neutral", "negative"], key=lambda x: sents.count(x))
        summary.append({"Aspect": a, "Positive": pos, "Neutral": neu, "Negative": neg, "Total": total, "Dominant": dom})
    return pd.DataFrame(summary).sort_values("Total", ascending=False), aspect_table


# ---------- Scraper ----------
def scrape_trustpilot(domain):
    base = f"https://www.trustpilot.com/review/{domain.strip()}"
    all_reviews, page = [], 1
    progress = st.empty()

    while True:
        url = f"{base}?page={page}"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            break
        soup = BeautifulSoup(response.text, "html.parser")
        sections = soup.find_all("section", class_="styles_reviewContentwrapper__K2aRu")
        if not sections:
            break

        for s in sections:
            title = s.find("h2").get_text(strip=True) if s.find("h2") else ""
            rev = s.find("p").get_text(strip=True) if s.find("p") else ""
            date = s.find("div", {"data-testid": "review-badge-date"})
            date = date.get_text(strip=True) if date else ""
            rating_div = s.find("div", {"data-service-review-rating": True})
            rating = rating_div["data-service-review-rating"] if rating_div else ""
            if rev:
                all_reviews.append({"rating": rating, "title": title, "review": rev, "date": date})
        progress.info(f"📄 Scraped page {page} ({len(all_reviews)} total)")
        page += 1
        time.sleep(1.2)

    progress.empty()
    return pd.DataFrame(all_reviews)


# ---------- UI ----------
st.title("💬 Trustpilot Aspect-Based Sentiment Analyzer (ABSA)")
st.write("Scrape and analyze Trustpilot reviews using aspect-based sentiment analysis in real-time.")

domain = st.text_input("🌐 Enter domain:", "www.facebook.com")

if st.button("🚀 Analyze"):
    with st.spinner(f"Scraping {domain} ..."):
        df = scrape_trustpilot(domain)

    if df.empty:
        st.error("⚠️ No reviews found!")
    else:
        st.success(f"✅ Collected {len(df)} reviews.")
        st.subheader("📄 Raw Scraped Data")
        st.dataframe(df)

        with st.spinner("🔍 Analyzing aspects..."):
            aspect_df, aspect_table = analyze_aspects(df)

        # Store in session state to prevent restart
        st.session_state.df = df
        st.session_state.aspect_df = aspect_df
        st.session_state.aspect_table = aspect_table

if 'aspect_df' in st.session_state:
    df = st.session_state.df
    aspect_df = st.session_state.aspect_df
    aspect_table = st.session_state.aspect_table

    st.subheader("🔍 Aspect Summary")
    st.dataframe(aspect_df)

    # ---------- Aspect Table per Review ----------
    st.subheader("🧩 Aspects & Opinions (Inline View)")

    num_reviews = st.slider("Select how many random reviews to show:", 1, min(10, len(df)), 5, key="review_slider")
    selected_indices = random.sample(range(len(df)), num_reviews)

    for idx in selected_indices:
        row = df.iloc[idx]
        st.markdown(f"**Review {idx+1}:** {row.review}")

        pairs = aspect_table[idx]
        if not pairs:
            st.caption("No aspect-opinion pairs found.")
        else:
            # Create colored boxes for opinions only
            colored_boxes = []
            for aspect, opinion, sentiment in pairs:
                color = {
                    "positive": "#4CAF50", 
                    "neutral": "#95A5A6", 
                    "negative": "#E74C3C"
                }.get(sentiment, "#95A5A6")
                
                box_html = f"""
                <span style='
                    display: inline-block;
                    background-color: {color};
                    color: white;
                    padding: 2px 8px;
                    margin: 2px;
                    border-radius: 4px;
                    font-size: 0.9em;
                '>{opinion}</span>
                """
                colored_boxes.append(f"{aspect} {box_html},")
            
            st.markdown("**Aspects & Opinions:** " + " ".join(colored_boxes), unsafe_allow_html=True)

    # ---------- Pie chart ----------
    st.subheader("📊 Sentiment Distribution")
    labels = ["Positive", "Neutral", "Negative"]
    sizes = [aspect_df["Positive"].sum(), aspect_df["Neutral"].sum(), aspect_df["Negative"].sum()]
    fig, ax = plt.subplots(figsize=(2, 2))
    fig.patch.set_alpha(0.0)  # transparent background
    ax.set_facecolor("none")
    
    # Set chart labels to white
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=20, 
           textprops={"fontsize": 5, "color": "white"})
    st.pyplot(fig)

    # ---------- Overall sentiment ----------
    pos, neu, neg = sizes
    total = pos + neu + neg
    score = (pos - neg) / total if total else 0
    overall = "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"
    color = {"positive": "#4CAF50", "neutral": "#95A5A6", "negative": "#E74C3C"}[overall]

    st.markdown(
        f"<div style='text-align:center;padding:16px;border-radius:8px;"
        f"background:{color};color:white;max-width:400px;margin:auto;'>"
        f"<b>OVERALL SENTIMENT: {overall.upper()}</b><br>(Score {score:.2f})</div>",
        unsafe_allow_html=True,
    )
