import os
import re
import streamlit as st
import textrazor
import pandas as pd
import altair as alt
from groq import Groq

# --- Setup ---

textrazor.api_key = "9a59e0c22ec2d995fa459cf0d61ebe07580bfaa426d1cf6dee35959e"
client = textrazor.TextRazor(extractors=[
    "entities",
    "topics",
    "words",
    "phrases",
    "dependency-trees",
    "relations",
    "entailments",
    "categories",
    "senses",
    "spelling",
    "translation"
])

GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or "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"
groq_client = Groq(api_key=GROQ_API_KEY)

# --- Helper Functions ---

def groq_ai_request(prompt: str) -> str:
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            stream=False,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Groq API Error: {e}"

def analyze_text(text: str):
    response = client.analyze(text)

    seo_keywords = []
    entities = []
    topics = []
    categories = []
    spelling_suggestions = []

    for entity in response.entities():
        entities.append({
            "id": entity.id,
            "relevance": entity.relevance_score,
            "confidence": entity.confidence_score,
        })
        seo_keywords.append({"keyword": entity.id, "relevance": entity.relevance_score})

    for topic in response.topics():
        topics.append({"label": topic.label, "score": topic.score})

    for cat in response.categories():
        categories.append({"label": cat.label, "score": cat.score})

    for word in response.words():
        if word.spelling_suggestions:
            spelling_suggestions.append({
                "token": word.token,
                "suggestions": word.spelling_suggestions
            })

    return {
        "seo_keywords": seo_keywords,
        "entities": entities,
        "topics": topics,
        "categories": categories,
        "spelling_suggestions": spelling_suggestions
    }

def get_recommended_keywords(keywords, threshold=0.2):
    return [kw["keyword"] for kw in keywords if kw["relevance"] >= threshold]

def insert_keywords(text, keywords):
    text_lower = text.lower()
    present = [kw.lower() for kw in keywords if re.search(r'\b' + re.escape(kw.lower()) + r'\b', text_lower)]
    to_add = [kw for kw in keywords if kw.lower() not in present]

    if not to_add:
        return text, False

    # Insert keywords before the last period or at the end if none found
    insertion_point = text.rfind(".")
    if insertion_point == -1:
        insertion_point = len(text)

    keywords_phrase = ", ".join(to_add)
    # Make sure to add a comma or "including" in a natural way
    new_text = (
        text[:insertion_point].rstrip()
        + f" including {keywords_phrase}"
        + text[insertion_point:]
    )
    return new_text, True

def get_keyword_snippets(text, keywords, window=30):
    """
    Extract snippets from text that include each keyword with a context window.
    """
    text_lower = text.lower()
    snippets = []
    for kw in keywords:
        pattern = re.compile(r'.{0,%d}\b%s\b.{0,%d}' % (window, re.escape(kw.lower()), window), re.IGNORECASE)
        match = pattern.search(text_lower)
        if match:
            snippet = match.group(0).strip()
            snippets.append(snippet)
    return snippets

# --- Streamlit App ---

st.set_page_config(
    page_title="🔥 SEO Analyzer & AI Enhancer 🔥",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚀"
)

st.markdown(
    """
    <style>
    /* Gradient background for header */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .header {
        font-size: 3rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 0.25rem;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    }
    .subheader {
        font-size: 1.3rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
        color: #dcd6f7;
    }
    .stTextArea>div>textarea {
        font-size: 1.2rem;
        font-weight: 500;
        border-radius: 12px;
        border: 2px solid #764ba2;
        padding: 1rem;
        min-height: 150px;
        color: #222;
    }
    .btn-primary {
        background: linear-gradient(90deg, #ff7e5f, #feb47b);
        border: none;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        box-shadow: 0 4px 10px rgba(255,126,95,0.6);
        cursor: pointer;
        transition: all 0.3s ease;
        color: white !important;
    }
    .btn-primary:hover {
        box-shadow: 0 6px 20px rgba(255,126,95,0.9);
        transform: scale(1.05);
    }
    .card {
        background: white;
        border-radius: 20px;
        padding: 1rem 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        color: #333;
        white-space: pre-wrap;
    }
    .badge {
        background: #764ba2;
        color: white;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-weight: 600;
        margin-right: 6px;
        display: inline-block;
    }
    .highlight-snippet {
        background: #fffbdd;
        border-left: 4px solid #ff7e5f;
        padding: 0.75rem 1rem;
        margin-bottom: 1rem;
        font-style: italic;
        color: #5a4d41;
        border-radius: 10px;
        white-space: pre-wrap;
    }
    details summary {
        font-weight: 700;
        font-size: 1.1rem;
        cursor: pointer;
        margin-bottom: 0.5rem;
    }
    details {
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="header">🔥 SEO Text Analyzer & AI Enhancer 🔥</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Paste your text and unlock powerful SEO insights with AI magic! ✨</div>', unsafe_allow_html=True)

user_text = st.text_area("🚀 Enter your text here:", height=220, placeholder="Paste your SEO content or article here...")

analyze_button = st.button("✨ Analyze & Enhance ✨", help="Click to analyze and get AI suggestions", key="analyze")

if analyze_button:
    if not user_text.strip():
        st.warning("⚠️ Please enter some text before analyzing!")
    else:
        with st.spinner("🛠️ Running analysis, hang tight..."):
            analysis = analyze_text(user_text)

        st.markdown("---")

        # Columns for charts + info
        col1, col2 = st.columns([2, 3])

        with col1:
            st.markdown("### 🏷️ Entities by Relevance")
            if analysis["entities"]:
                entities_df = pd.DataFrame([
                    {"Entity": e["id"], "Relevance": e["relevance"]}
                    for e in analysis["entities"]
                ])
                chart = alt.Chart(entities_df).mark_bar(color="#764ba2").encode(
                    x=alt.X('Relevance:Q', scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y('Entity:N', sort='-x'),
                    tooltip=['Entity', 'Relevance']
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No entities detected.")

            st.markdown("### 🎯 Topics by Score")
            if analysis["topics"]:
                topics_df = pd.DataFrame([
                    {"Topic": t["label"], "Score": t["score"]}
                    for t in analysis["topics"]
                ])
                chart = alt.Chart(topics_df).mark_bar(color="#ff7e5f").encode(
                    x=alt.X('Score:Q', scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y('Topic:N', sort='-x'),
                    tooltip=['Topic', 'Score']
                ).properties(height=260)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No topics detected.")

        with col2:
            st.markdown("### 🗂️ Categories by Score")
            if analysis["categories"]:
                categories_df = pd.DataFrame([
                    {"Category": c["label"], "Score": c["score"]}
                    for c in analysis["categories"]
                ])
                chart = alt.Chart(categories_df).mark_bar(color="#feb47b").encode(
                    x=alt.X('Score:Q', scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y('Category:N', sort='-x'),
                    tooltip=['Category', 'Score']
                ).properties(height=570)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No categories detected.")

        # SEO Keywords table + badges
        st.markdown("### 🔑 SEO Keywords and Relevance Scores")
        seo_keywords = analysis["seo_keywords"]
        if seo_keywords:
            seo_keywords_sorted = sorted(seo_keywords, key=lambda x: x["relevance"], reverse=True)
            keywords_df = pd.DataFrame([
                {"Keyword": kw["keyword"], "Relevance": kw["relevance"]}
                for kw in seo_keywords_sorted
            ])

            # Show keywords as badges
            badge_html = ""
            for kw in seo_keywords_sorted:
                badge_html += f'<span class="badge">{kw["keyword"]} ({kw["relevance"]:.2f})</span> '

            st.markdown(badge_html, unsafe_allow_html=True)
        else:
            st.info("No SEO keywords detected.")

        # Spelling Suggestions
        if analysis["spelling_suggestions"]:
            with st.expander("📝 Spelling Suggestions (Click to expand)"):
                for sug in analysis["spelling_suggestions"]:
                    st.write(f"**{sug['token']}** ➡ Suggestions: {', '.join(map(str, sug['suggestions']))}")

        # Recommended Keywords filtering
        recommended = get_recommended_keywords(seo_keywords, threshold=0.2)
        if recommended:
            st.markdown(f"### 💡 Recommended Keywords (Relevance ≥ 0.2):")
            recommended_html = ""
            for kw in recommended:
                recommended_html += f'<span class="badge" style="background:#ff7e5f">{kw}</span> '
            st.markdown(recommended_html, unsafe_allow_html=True)
        else:
            st.info("No recommended keywords found based on threshold.")

        # Insert recommended keywords into original text
        updated_text, inserted = insert_keywords(user_text, recommended)

        # Show text after keyword insertion in a card
        # Show original vs upgraded text side-by-side
        st.markdown("### 🔍 Keyword Enhancement Comparison")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("#### 📝 Original Text")
            st.markdown(f'<div class="card">{user_text}</div>', unsafe_allow_html=True)
        
        with col_b:
            st.markdown("#### 📝 Upgraded Text with Keywords")
            if inserted:
                st.success("Keywords successfully inserted into the text!")
            else:
                st.info("All recommended keywords were already present.")
            st.markdown(f'<div class="card">{updated_text}</div>', unsafe_allow_html=True)

        # Keyword snippet highlights
        snippets = get_keyword_snippets(updated_text, recommended) if inserted else []
        if inserted and snippets:
            st.markdown("### 🔍 Keyword Insertion Highlights")
            for snippet in snippets:
                st.markdown(f'<div class="highlight-snippet">…{snippet}…</div>', unsafe_allow_html=True)
        else:
            st.info("No keyword insertion snippets available.")

        # AI Prompt with positive only feedback
        groq_prompt = (
            f"Analyze the following text for SEO optimization and provide only positive feedback and praise.\n"
            f"Do NOT mention any problems or negative suggestions.\n"
            f"Show how the suggested keywords improve the text by giving a snippet with a few words before and after the inserted keywords.\n\n"
            f"{updated_text}\n\n"
            f"Also, suggest a positive meta description based on the text and recommended keywords: {', '.join(recommended)}"
        )

        st.markdown("---")
        st.markdown("### 🤖 AI SEO Improvement Suggestions")

        with st.spinner("Generating AI suggestions... 🌟"):
            groq_response = groq_ai_request(groq_prompt)

        st.text_area(
            "AI Suggestions & Meta Description:",
            value=groq_response,
            height=280,
            help="You can copy the AI-generated SEO improvement suggestions and meta description here."
        )

        # Bonus: share button (copy to clipboard style)
        st.markdown("""
        <style>
        .copy-button {
            background: linear-gradient(90deg, #ff7e5f, #feb47b);
            border: none;
            font-weight: 700;
            font-size: 1.1rem;
            padding: 0.5rem 1.5rem;
            border-radius: 50px;
            box-shadow: 0 4px 10px rgba(255,126,95,0.6);
            cursor: pointer;
            color: white;
            margin-top: 10px;
            user-select: none;
            transition: all 0.3s ease;
        }
        .copy-button:hover {
            box-shadow: 0 6px 20px rgba(255,126,95,0.9);
            transform: scale(1.05);
        }
        </style>
        <button class="copy-button" onclick="navigator.clipboard.writeText(document.querySelector('textarea').value)">📋 Copy AI Suggestions</button>
        """, unsafe_allow_html=True)
