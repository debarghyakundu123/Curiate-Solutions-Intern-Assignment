import os
import re
import html

import streamlit as st
import textrazor
import pandas as pd
import altair as alt
from groq import Groq
import webbrowser

# --- Setup ---

textrazor.api_key = st.secrets["textrazor_api_key"]
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

groq_client = Groq(api_key=st.secrets["groq_api_key"])

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
        return text, False, []

    insertion_point = text.rfind(".")
    if insertion_point == -1:
        insertion_point = len(text)

    keywords_phrase = ", ".join(to_add)
    new_text = (
        text[:insertion_point].rstrip()
        + f" including {keywords_phrase}"
        + text[insertion_point:]
    )
    return new_text, True, to_add


def get_keyword_snippets(text, keywords, window=30):
    text_lower = text.lower()
    snippets = []
    for kw in keywords:
        pattern = re.compile(r'.{0,%d}\b%s\b.{0,%d}' % (window, re.escape(kw.lower()), window), re.IGNORECASE)
        match = pattern.search(text_lower)
        if match:
            snippet = match.group(0)
            snippets.append(snippet.strip())
    return snippets

def highlight_inserted_keywords(text, keywords):
    """Wrap each keyword in <mark> tags for highlighting in HTML."""
    keywords = sorted(keywords, key=len, reverse=True)
    escaped_text = html.escape(text)
    safe_text = html.escape(text)

    
    for kw in keywords:
        kw_escaped = html.escape(kw)
        pattern = re.compile(rf'\b({re.escape(kw_escaped)})\b', re.IGNORECASE)
        escaped_text = pattern.sub(r'<mark>\1</mark>', escaped_text)
        safe_text = pattern.sub(f'<span style="background-color:#000;color:#fff;padding:0 4px;border-radius:4px;">{kw}</span>', safe_text)

    # Black background with light text and padding
    html_text = f"""
    <div style="
        height:300px; 
        overflow-y:auto; 
        font-family: monospace; 
        background:#000000; 
        color:#eeeeee; 
        padding:10px; 
        border-radius:8px;
        white-space: pre-wrap;
        ">
    {escaped_text}
    </div>
    """
    return html_text



# --- Streamlit App ---

st.set_page_config(
    page_title="🔥 SEO Analyzer & AI Enhancer 🔥",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚀"
)
# Paste LinkedIn section below this:
st.title("Welcome to the SEO Analyzer & AI Enhancer 🧠📈")
st.markdown("### 🚀 Supercharge Your Content with AI and SEO Insights")

st.markdown(
    """
    <a href="https://www.linkedin.com/in/debarghyakundu/" target="_blank">
        <button style="
            background-color: #0e76a8;
            color: white;
            padding: 0.7rem 1.5rem;
            font-size: 1rem;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        ">
            🔗 Visit My LinkedIn Profile
        </button>
    </a>
    """,
    unsafe_allow_html=True
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
        col1 = st.container()

        with col1:
            left_col, spacer_col, right_col = st.columns([5,1,5])  # Left and Right narrow, middle wide spacer
            
            
            with left_col:
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
                    ).properties(height=400)  # Bigger height for bigger display
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("No entities detected.")
        
        # Spacer column just empty for big space
        
            with right_col:
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
                    ).properties(height=400)  # Bigger height for bigger display
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("No topics detected.")


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
        updated_text, inserted, inserted_keywords = insert_keywords(user_text, recommended)
        st.markdown("### 🔄 Before vs After: Text Comparison")
        before_col, after_col = st.columns(2)
        with before_col:
            st.markdown("**📝 Original Text**")
            st.text_area("Original Text", value=user_text, height=300)  # adjust height as needed
        
        with after_col:
            st.markdown("**✅ Enhanced Text with Keywords**")
            if inserted and inserted_keywords:
                highlighted_text = highlight_inserted_keywords(updated_text, inserted_keywords)
                st.markdown(highlighted_text, unsafe_allow_html=True)
            else:
                st.text_area("Enhanced Text", value=updated_text, height=300)
        

        
        # Show text after keyword insertion in a card
        st.markdown("### ✍️ Text with Inserted Keywords")
        if inserted:
            st.success("Keywords successfully inserted into text:")
        else:
            st.info("All recommended keywords are already present in the text.")

        st.markdown(f'<div class="card">{updated_text}</div>', unsafe_allow_html=True)

        # Keyword snippet highlights
        snippets = get_keyword_snippets(updated_text, recommended) if inserted else []
        if inserted and snippets:
            st.markdown("### 🔍 Keyword Insertion Highlights")
            for i, snippet in enumerate(snippets):
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

        st.text_area("AI Suggestions & Meta Description:", value=groq_response, height=280)

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
