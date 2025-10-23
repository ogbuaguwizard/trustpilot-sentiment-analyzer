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
st.set_page_config(
    page_title="Trustpilot ABSA Analyzer", 
    page_icon="💬", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------- Custom CSS for GitHub Dark Theme ----------
st.markdown("""
<style>
    /* GitHub Dark Theme Variables */
    :root {
        --primary: #2ea043;
        --primary-dark: #238636;
        --secondary: #161b22;
        --accent: #58a6ff;
        --accent-purple: #8957e5;
        --text-primary: #f0f6fc;
        --text-secondary: #8b949e;
        --border: #30363d;
        --bg-dark: #0d1117;
        --bg-darker: #010409;
        --success: #3fb950;
        --warning: #d29922;
        --danger: #f85149;
        --card-shadow: 0 8px 24px rgba(1, 4, 9, 0.5);
        --transition: all 0.2s cubic-bezier(0.3, 0, 0.5, 1);
    }

    /* Main Background */
    .stApp {
        background-color: var(--bg-dark);
        color: var(--text-primary);
        font-family: 'Segoe UI', system-ui, sans-serif;
    }

    /* Headers */
    h1, h2, h3 {
        color: var(--accent) !important;
        font-weight: 600 !important;
        border-bottom: 1px solid var(--border);
        padding-bottom: 0.5rem;
        margin-bottom: 1rem !important;
    }

    h1 {
        font-size: 2.5rem !important;
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-purple) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        border-bottom: 2px solid var(--border);
    }

    /* Cards and Containers */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    .stDataFrame {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: var(--transition) !important;
        width: 100%;
    }

    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(46, 160, 67, 0.3) !important;
    }

    /* Text Input */
    .stTextInput input {
        background-color: var(--bg-darker) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        padding: 0.75rem !important;
        font-size: 1rem !important;
    }

    .stTextInput input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.2) !important;
    }

    /* Labels */
    .stTextInput label {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
    }

    /* Success/Error Messages */
    .stAlert {
        border-radius: 8px !important;
        border: 1px solid var(--border) !important;
        background-color: var(--secondary) !important;
    }

    .stAlert [data-testid="stMarkdownContainer"] {
        color: var(--text-primary) !important;
    }

    /* Dataframes */
    .dataframe {
        background-color: var(--secondary) !important;
        color: var(--text-primary) !important;
    }

    .dataframe thead th {
        background-color: var(--bg-darker) !important;
        color: var(--accent) !important;
        border-bottom: 2px solid var(--border) !important;
    }

    .dataframe tbody tr {
        background-color: var(--secondary) !important;
    }

    .dataframe tbody tr:nth-child(even) {
        background-color: var(--bg-darker) !important;
    }

    .dataframe td {
        border-color: var(--border) !important;
        color: var(--text-primary) !important;
    }

    /* Slider */
    .stSlider {
        padding: 1rem 0;
    }

    .stSlider [data-testid="stThumbValue"] {
        color: var(--accent) !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-color: var(--accent) transparent transparent transparent !important;
    }

    /* Custom Cards */
    .custom-card {
        background-color: var(--secondary);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--card-shadow);
        transition: var(--transition);
    }

    .custom-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(1, 4, 9, 0.6);
    }

    /* Review Cards */
    .review-card {
        background-color: var(--secondary);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: var(--transition);
    }

    .review-card:hover {
        border-color: var(--accent);
        transform: translateY(-1px);
    }

    /* Aspect Tags */
    .aspect-tag {
        display: inline-block;
        background-color: var(--bg-darker);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 0.25rem 0.5rem;
        margin: 0.25rem;
        font-size: 0.85rem;
        color: var(--text-secondary);
    }

    .opinion-positive {
        background-color: var(--success) !important;
        color: white !important;
        border: none !important;
    }

    .opinion-neutral {
        background-color: var(--warning) !important;
        color: white !important;
        border: none !important;
    }

    .opinion-negative {
        background-color: var(--danger) !important;
        color: white !important;
        border: none !important;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background-color: var(--secondary);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
    }

    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
    }

    [data-testid="stMetricDelta"] {
        color: var(--text-secondary) !important;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom Header */
    .main-header {
        background: linear-gradient(135deg, var(--bg-dark) 0%, var(--bg-darker) 100%);
        border-bottom: 1px solid var(--border);
        padding: 2rem 0;
        margin-bottom: 2rem;
        border-radius: 0 0 12px 12px;
    }

    /* Status Indicators */
    .status-positive {
        color: var(--success);
        font-weight: 600;
    }

    .status-neutral {
        color: var(--warning);
        font-weight: 600;
    }

    .status-negative {
        color: var(--danger);
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

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

# ---------- Modern UI Layout ----------

# Header Section
st.markdown("""
<div class="main-header">
    <div style="text-align: center;">
        <h1 style="border: none; padding: 0; margin: 0;">💬 Trustpilot ABSA Analyzer</h1>
        <p style="color: #8b949e; font-size: 1.2rem; margin: 0.5rem 0 0 0;">
            Aspect-Based Sentiment Analysis with GitHub Dark Theme
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Content
col1, col2 = st.columns([2, 1])

with col1:
    domain = st.text_input(
        "🌐 **Enter Trustpilot Domain:**", 
        "www.facebook.com",
        help="Enter the domain name as it appears on Trustpilot (e.g., 'www.facebook.com')"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_clicked = st.button("🚀 **Start Analysis**", use_container_width=True)

# Analysis Section
if analyze_clicked:
    with st.spinner(f"🔍 **Scraping reviews from {domain}...**"):
        df = scrape_trustpilot(domain)

    if df.empty:
        st.error("""
        ⚠️ **No reviews found!** 
        - Please check the domain name
        - Ensure the company has Trustpilot reviews
        - Try a different domain
        """)
    else:
        st.success(f"✅ **Successfully collected {len(df)} reviews**")
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Reviews", len(df))
        with col2:
            avg_rating = pd.to_numeric(df['rating'], errors='coerce').mean()
            st.metric("Average Rating", f"{avg_rating:.1f} ⭐" if not pd.isna(avg_rating) else "N/A")
        with col3:
            st.metric("Data Collection", "Complete", delta="100%")

        # Raw Data Section
        with st.expander("📄 **Raw Scraped Data**", expanded=False):
            st.dataframe(df, use_container_width=True)

        # Aspect Analysis
        with st.spinner("🔍 **Analyzing aspects and sentiments...**"):
            aspect_df, aspect_table = analyze_aspects(df)

        # Store in session state
        st.session_state.df = df
        st.session_state.aspect_df = aspect_df
        st.session_state.aspect_table = aspect_table

# Display Results if available
if 'aspect_df' in st.session_state:
    df = st.session_state.df
    aspect_df = st.session_state.aspect_df
    aspect_table = st.session_state.aspect_table

    # Aspect Summary
    st.markdown("### 📊 Aspect Sentiment Summary")
    st.dataframe(aspect_df, use_container_width=True)

    # Visualizations
    col1, col2 = st.columns([2, 1])

    with col1:
        # Sentiment Distribution Bar Chart
        st.markdown("### 📈 Sentiment Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')
        
        aspects = aspect_df.head(10)['Aspect']
        pos = aspect_df.head(10)['Positive']
        neu = aspect_df.head(10)['Neutral']
        neg = aspect_df.head(10)['Negative']
        
        ax.barh(aspects, pos, label='Positive', color='#3fb950')
        ax.barh(aspects, neu, left=pos, label='Neutral', color='#d29922')
        ax.barh(aspects, neg, left=pos+neu, label='Negative', color='#f85149')
        
        ax.set_xlabel('Count', color='#f0f6fc')
        ax.set_ylabel('Aspects', color='#f0f6fc')
        ax.tick_params(colors='#8b949e')
        ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#f0f6fc')
        ax.spines['bottom'].set_color('#30363d')
        ax.spines['top'].set_color('#30363d')
        ax.spines['right'].set_color('#30363d')
        ax.spines['left'].set_color('#30363d')
        
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        # Overall Sentiment Pie Chart
        st.markdown("### 🥧 Overall Sentiment")
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')
        
        labels = ["Positive", "Neutral", "Negative"]
        sizes = [aspect_df["Positive"].sum(), aspect_df["Neutral"].sum(), aspect_df["Negative"].sum()]
        colors = ['#3fb950', '#d29922', '#f85149']
        
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct="%1.1f%%", startangle=90,
            colors=colors, textprops={'color': '#f0f6fc', 'fontsize': 10}
        )
        
        plt.setp(autotexts, size=10, weight="bold", color='#0d1117')
        st.pyplot(fig)

    # Review Analysis Section
    st.markdown("### 🧩 Detailed Review Analysis")
    
    num_reviews = st.slider(
        "**Select number of reviews to analyze:**", 
        1, min(10, len(df)), 5, key="review_slider"
    )
    
    selected_indices = random.sample(range(len(df)), num_reviews)

    for idx in selected_indices:
        row = df.iloc[idx]
        
        # Create a custom card for each review
        st.markdown(f"""
        <div class="review-card">
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.5rem;">
                <strong style="color: #58a6ff;">Review {idx+1}</strong>
                <small style="color: #8b949e;">{row.date} • Rating: {row.rating} ⭐</small>
            </div>
            <div style="color: #f0f6fc; font-style: italic; margin-bottom: 0.5rem;">"{row.review}"</div>
        """, unsafe_allow_html=True)
        
        pairs = aspect_table[idx]
        if not pairs:
            st.markdown('<small style="color: #8b949e;">No aspect-opinion pairs detected</small>', unsafe_allow_html=True)
        else:
            st.markdown("<div style='margin-top: 0.5rem;'><strong>Detected Aspects & Opinions:</strong></div>", unsafe_allow_html=True)
            
            # Create aspect-opinion pairs with colored sentiment tags
            aspect_html = []
            for aspect, opinion, sentiment in pairs:
                sentiment_class = f"opinion-{sentiment}"
                aspect_html.append(f"""
                    <span class="aspect-tag">
                        <strong>{aspect}</strong>: 
                        <span class="aspect-tag {sentiment_class}">{opinion}</span>
                    </span>
                """)
            
            st.markdown("".join(aspect_html), unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Overall Sentiment Score
    st.markdown("### 🎯 Overall Analysis Summary")
    
    pos, neu, neg = aspect_df["Positive"].sum(), aspect_df["Neutral"].sum(), aspect_df["Negative"].sum()
    total = pos + neu + neg
    score = (pos - neg) / total if total else 0
    overall = "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"
    
    sentiment_color = {
        "positive": "#3fb950", 
        "neutral": "#d29922", 
        "negative": "#f85149"
    }[overall]
    
    # Create a beautiful sentiment card
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, {sentiment_color}20, {sentiment_color}10);
        border: 2px solid {sentiment_color};
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    '>
        <h2 style='color: {sentiment_color}; margin: 0; border: none;'>Overall Sentiment: {overall.upper()}</h2>
        <p style='color: #8b949e; font-size: 1.2rem; margin: 0.5rem 0 0 0;'>
            Sentiment Score: <strong style='color: {sentiment_color};'>{score:.3f}</strong>
        </p>
        <div style='display: flex; justify-content: center; gap: 2rem; margin-top: 1rem;'>
            <span style='color: #3fb950;'>👍 Positive: {pos}</span>
            <span style='color: #d29922;'>⚖️ Neutral: {neu}</span>
            <span style='color: #f85149;'>👎 Negative: {neg}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #8b949e; padding: 2rem 0;">
    <p>
        Made with ❤️ by <strong style="color: #58a6ff;">Ogbuaguwizard</strong>
    </p>
    <small>© 2025 Trustpilot ABSA Analyzer • MIT License</small>
</div>
""", unsafe_allow_html=True)