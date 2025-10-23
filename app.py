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
        --font-primary: 'Segoe UI', system-ui, sans-serif;
        --font-mono: 'SF Mono', Monaco, 'Cascadia Code', monospace;
    }

    /* Main Background */
    .stApp {
        background-color: var(--bg-dark);
        color: var(--text-primary);
        font-family: var(--font-primary);
        line-height: 1.6;
    }

    /* GitHub Corner Style */
    .github-corner {
        position: absolute;
        top: 0;
        right: 0;
        border: 0;
        z-index: 1000;
    }

    /* Header Styles */
    .header-gradient {
        background: linear-gradient(135deg, var(--bg-dark) 0%, var(--bg-darker) 100%);
        border-bottom: 1px solid var(--border);
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 2rem;
    }

    .brand {
        font-family: var(--font-mono);
        font-weight: 600;
        color: var(--accent);
        font-size: 2.5rem;
        letter-spacing: -0.5px;
        margin: 0;
        border: none !important;
        padding: 0 !important;
    }

    .brand span {
        color: var(--primary);
    }

    .subtitle {
        color: var(--text-secondary);
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
    }

    /* Card Styles */
    .modern-card {
        background-color: var(--secondary);
        border: 1px solid var(--border);
        border-radius: 12px;
        box-shadow: var(--card-shadow);
        transition: var(--transition);
        overflow: hidden;
        margin-bottom: 1.5rem;
    }

    .modern-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(1, 4, 9, 0.6);
    }

    .card-header {
        background-color: rgba(1, 4, 9, 0.3);
        border-bottom: 1px solid var(--border);
        padding: 1rem 1.5rem;
    }

    .card-title {
        font-weight: 600;
        font-size: 1.2rem;
        margin: 0;
        color: var(--accent);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .card-body {
        padding: 1.5rem;
    }

    /* Button Styles */
    .stButton button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: var(--transition) !important;
        width: 100%;
        font-family: var(--font-primary) !important;
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
        font-family: var(--font-mono) !important;
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

    /* Metrics */
    [data-testid="stMetric"] {
        background-color: var(--secondary);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        height: 100%;
    }

    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-size: 0.9rem !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 2rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricDelta"] {
        color: var(--text-secondary) !important;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Progress Bar */
    .progress-modern {
        background-color: var(--bg-darker);
        border: 1px solid var(--border);
        border-radius: 8px;
        height: 8px;
        overflow: hidden;
        margin-bottom: 0.5rem;
    }

    .progress-bar-modern {
        background: linear-gradient(90deg, var(--primary) 0%, var(--accent) 100%);
        height: 100%;
        transition: width 0.3s ease;
    }

    .progress-steps {
        display: flex;
        justify-content: space-between;
        margin-top: 0.5rem;
        font-size: 0.85rem;
        color: var(--text-secondary);
    }

    .progress-step {
        text-align: center;
        flex: 1;
    }

    .progress-step.active {
        color: var(--accent);
        font-weight: 600;
    }

    /* Chart Containers */
    .chart-container-modern {
        background-color: var(--secondary);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--card-shadow);
    }

    .chart-title {
        color: var(--accent);
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 1.2rem;
    }

    /* Aspect Table */
    .aspect-table-container {
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid var(--border);
        border-radius: 8px;
    }

    /* Review Cards */
    .review-card {
        background-color: var(--secondary);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
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

    /* Badge */
    .badge-modern {
        background-color: var(--accent-purple);
        color: white;
        font-weight: 500;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-size: 0.9rem;
    }

    /* Footer */
    .footer {
        border-top: 1px solid var(--border);
        padding: 1.5rem 0;
        margin-top: 3rem;
        text-align: center;
    }

    .font-mono {
        font-family: var(--font-mono);
    }

    /* Collapsible Section */
    .collapsible-section {
        margin-bottom: 1.5rem;
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

# GitHub Corner
st.markdown("""
<a href="https://github.com/your-repo/trustpilot-absa" class="github-corner" aria-label="View source on GitHub">
    <svg width="80" height="80" viewBox="0 0 250 250" style="fill:#2ea043; color:#0d1117; position: absolute; top: 0; border: 0; right: 0;" aria-hidden="true">
        <path d="M0,0 L115,115 L130,115 L142,142 L250,250 L250,0 Z"></path>
        <path d="M128.3,109.0 C113.8,99.7 119.0,89.6 119.0,89.6 C122.0,82.7 120.5,78.6 120.5,78.6 C119.2,72.0 123.4,76.3 123.4,76.3 C127.3,80.9 125.5,87.3 125.5,87.3 C122.9,97.6 130.6,101.9 134.4,103.2" fill="currentColor" style="transform-origin: 130px 106px;" class="octo-arm"></path>
        <path d="M115.0,115.0 C114.9,115.1 118.7,116.5 119.8,115.4 L133.7,101.6 C136.9,99.2 139.9,98.4 142.2,98.6 C133.8,88.0 127.5,74.4 143.8,58.0 C148.5,53.4 154.0,51.2 159.7,51.0 C160.3,49.4 163.2,43.6 171.4,40.1 C171.4,40.1 176.1,42.5 178.8,56.2 C183.1,58.6 187.2,61.8 190.9,65.4 C194.5,69.0 197.7,73.2 200.1,77.6 C213.8,80.2 216.3,84.9 216.3,84.9 C212.7,93.1 206.9,96.0 205.4,96.6 C205.1,102.4 203.0,107.8 198.3,112.5 C181.9,128.9 168.3,122.5 157.7,114.1 C157.9,116.9 156.7,120.9 152.7,124.9 L141.0,136.5 C139.8,137.7 141.6,141.9 141.8,141.8 Z" fill="currentColor" class="octo-body"></path>
    </svg>
</a>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="header-gradient">
    <div class="row align-items-center">
        <div class="col-md-8">
            <h1 class="brand">
                &lt;<span>Trustpilot ABSA</span>/&gt;
            </h1>
            <p class="subtitle">Advanced Aspect-Based Sentiment Analysis for Trustpilot Reviews</p>
        </div>
        <div class="col-md-4 text-md-end mt-3 mt-md-0">
            <span class="badge-modern">v2.0</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Input Section
st.markdown("""
<div class="modern-card">
    <div class="card-header">
        <h2 class="card-title">
            <i class="bi bi-search"></i>
            Analyze Trustpilot Reviews
        </h2>
    </div>
    <div class="card-body">
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    domain = st.text_input(
        "**Enter Trustpilot Domain:**", 
        "www.facebook.com",
        placeholder="Enter Trustpilot domain (e.g., www.facebook.com)",
        help="Enter the domain name as it appears on Trustpilot (e.g., 'www.facebook.com')",
        label_visibility="collapsed"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_clicked = st.button("🚀 **Start Analysis**", use_container_width=True)

st.markdown("""
<small class="text-muted-modern">
    <i class="bi bi-info-circle me-1"></i>Enter the domain name as it appears on Trustpilot (e.g., 'www.facebook.com')
</small>
""", unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)

# Progress Section (initially hidden)
progress_section = st.empty()

# Analysis Section
if analyze_clicked:
    with progress_section.container():
        st.markdown("""
        <div class="modern-card">
            <div class="card-header">
                <h2 class="card-title">
                    <i class="bi bi-graph-up"></i>
                    Analysis Progress
                </h2>
            </div>
            <div class="card-body">
        """, unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate progress updates
        for percent_complete in range(0, 101, 20):
            if percent_complete == 0:
                status_text.text("🔄 Starting analysis...")
            elif percent_complete == 20:
                status_text.text("🌐 Connecting to Trustpilot...")
            elif percent_complete == 40:
                status_text.text("📄 Scraping reviews...")
            elif percent_complete == 60:
                status_text.text("🔍 Analyzing aspects and sentiment...")
            elif percent_complete == 80:
                status_text.text("📊 Generating visualizations...")
            elif percent_complete == 100:
                status_text.text("✅ Analysis complete!")
            
            progress_bar.progress(percent_complete)
            time.sleep(0.5)

        st.markdown("</div></div>", unsafe_allow_html=True)

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
        # Clear progress section
        progress_section.empty()
        
        # Success Alert
        st.success(f"✅ **Successfully collected {len(df)} reviews from {domain}**")
        
        # Metrics Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Reviews", len(df))
        with col2:
            avg_rating = pd.to_numeric(df['rating'], errors='coerce').mean()
            st.metric("Average Rating", f"{avg_rating:.1f} ⭐" if not pd.isna(avg_rating) else "N/A")
        with col3:
            st.metric("Aspects Found", "Calculating...")
        with col4:
            st.metric("Sentiment Score", "0.0")

        # Store initial data
        st.session_state.df = df

        # Aspect Analysis
        with st.spinner("🔍 **Analyzing aspects and sentiments...**"):
            aspect_df, aspect_table = analyze_aspects(df)

        # Update metrics
        with col3:
            st.metric("Aspects Found", len(aspect_df))
        with col4:
            pos, neu, neg = aspect_df["Positive"].sum(), aspect_df["Neutral"].sum(), aspect_df["Negative"].sum()
            total = pos + neu + neg
            sentiment_score = (pos - neg) / total if total else 0
            st.metric("Sentiment Score", f"{sentiment_score:.3f}")

        # Store in session state
        st.session_state.aspect_df = aspect_df
        st.session_state.aspect_table = aspect_table
        st.session_state.sentiment_score = sentiment_score

# Display Results if available
if 'aspect_df' in st.session_state:
    df = st.session_state.df
    aspect_df = st.session_state.aspect_df
    aspect_table = st.session_state.aspect_table
    sentiment_score = st.session_state.sentiment_score

    # Charts Section
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="chart-container-modern">
            <h3 class="chart-title">
                <i class="bi bi-bar-chart"></i>
                Aspect Sentiment Distribution
            </h3>
        """, unsafe_allow_html=True)
        
        # Sentiment Distribution Bar Chart
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
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="chart-container-modern">
            <h3 class="chart-title">
                <i class="bi bi-pie-chart"></i>
                Overall Sentiment
            </h3>
        """, unsafe_allow_html=True)
        
        # Overall Sentiment Pie Chart
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
        st.markdown("</div>", unsafe_allow_html=True)

    # Aspect Details Table (Collapsible)
    with st.expander("📊 **Aspect Sentiment Summary**", expanded=True):
        st.markdown("""
        <div class="aspect-table-container">
        """, unsafe_allow_html=True)
        st.dataframe(aspect_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Raw Data Section (Collapsible)
    with st.expander("📄 **Raw Scraped Data**", expanded=False):
        st.dataframe(df, use_container_width=True)

    # Review Analysis Section
    st.markdown("""
    <div class="modern-card">
        <div class="card-header">
            <h2 class="card-title">
                <i class="bi bi-chat-left-text"></i>
                Detailed Review Analysis
            </h2>
        </div>
        <div class="card-body">
    """, unsafe_allow_html=True)

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
                <small style="color: #8b949e;">
                    <i class="bi bi-calendar me-1"></i>{row.date} 
                    <i class="bi bi-star ms-2 me-1"></i>{row.rating} ⭐
                </small>
            </div>
            <div style="color: #f0f6fc; font-style: italic; margin-bottom: 0.5rem;">"{row.review}"</div>
        """, unsafe_allow_html=True)
        
        pairs = aspect_table[idx]
        if not pairs:
            st.markdown('<small style="color: #8b949e;"><i class="bi bi-info-circle me-1"></i>No aspect-opinion pairs detected</small>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="margin-top: 0.5rem;"><strong style="color: #8b949e;"><i class="bi bi-tags me-1"></i>Detected Aspects & Opinions:</strong></div>', unsafe_allow_html=True)
            
            # Create aspect-opinion pairs with colored sentiment tags
            aspect_html = []
            for aspect, opinion, sentiment in pairs:
                sentiment_class = f"opinion-{sentiment}"
                icon = "emoji-smile" if sentiment == "positive" else "emoji-neutral" if sentiment == "neutral" else "emoji-frown"
                aspect_html.append(f"""
                    <span class="aspect-tag">
                        <strong>{aspect}</strong>: 
                        <span class="aspect-tag {sentiment_class}">
                            <i class="bi bi-{icon} me-1"></i>{opinion}
                        </span>
                    </span>
                """)
            
            st.markdown("".join(aspect_html), unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

    # Overall Sentiment Banner
    pos, neu, neg = aspect_df["Positive"].sum(), aspect_df["Neutral"].sum(), aspect_df["Negative"].sum()
    overall = "positive" if sentiment_score > 0.05 else "negative" if sentiment_score < -0.05 else "neutral"
    
    sentiment_color = {
        "positive": "#3fb950", 
        "neutral": "#d29922", 
        "negative": "#f85149"
    }[overall]
    
    icon = "emoji-smile" if overall == "positive" else "emoji-neutral" if overall == "neutral" else "emoji-frown"

    # Create overall sentiment banner
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, {sentiment_color}20, {sentiment_color}10);
        border: 2px solid {sentiment_color};
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    '>
        <h2 style='color: {sentiment_color}; margin: 0; border: none;'>
            <i class="bi bi-{icon} me-2"></i>Overall Sentiment: {overall.upper()}
        </h2>
        <p style='color: #8b949e; font-size: 1.2rem; margin: 0.5rem 0 0 0;'>
            Sentiment Score: <strong style='color: {sentiment_color};'>{sentiment_score:.3f}</strong>
        </p>
        <div style='display: flex; justify-content: center; gap: 2rem; margin-top: 1rem;'>
            <span style='color: #3fb950;'><i class="bi bi-arrow-up-circle"></i> Positive: {pos}</span>
            <span style='color: #d29922;'><i class="bi bi-dash-circle"></i> Neutral: {neu}</span>
            <span style='color: #f85149;'><i class="bi bi-arrow-down-circle"></i> Negative: {neg}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p class="text-muted-modern mb-2 font-mono">Made with <i class="bi bi-heart-fill text-danger"></i> by Ogbuaguwizard</p>
    <p class="small text-muted-modern font-mono">© 2025 Trustpilot ABSA Analyzer</p>
</div>
""", unsafe_allow_html=True)