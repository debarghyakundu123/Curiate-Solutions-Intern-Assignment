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
    new_text = (
        text[:insertion_point].rstrip()
        + f" including {keywords_phrase}"
        + text[insertion_point:]
    )
    return new_text, True

def get_keyword_snippets(text, keywords, window=30):
    text_lower = text.lower()
    snippets = []
    for kw in keywords:
        pattern = re.compile(r'.{0,%d}\b%s\b.{0,%d}' % (window, re.escape(kw.lower()), window), re.IGNORECASE)
        match = pattern.search(text_lower)
        if match:
            snippet = match.group(0).strip()
            snippets.append(snippet)
    return snippets

def compare_items(original, enhanced, key='id', score='relevance'):
    orig_map = {e[key]: e[score] for e in original}
    enh_map = {e[key]: e[score] for e in enhanced}
    all_items = set(orig_map) | set(enh_map)
    data = []
    for item in all_items:
        data.append({
            'Item': item,
            'Original Score': orig_map.get(item, 0),
            'Enhanced Score': enh_map.get(item, 0)
        })
    return pd.DataFrame(data)

def grouped_bar_chart(df, title, item_col='Item'):
    if df.empty:
        return None
    chart = alt.Chart(df).transform_fold(
        ['Original Score', 'Enhanced Score'],
        as_=['Version', 'Score']
    ).mark_bar().encode(
        x=alt.X('Version:N', title=None),
        y=alt.Y('Score:Q', scale=alt.Scale(domain=[0, 1])),
        color='Version:N',
        column=alt.Column(f'{item_col}:N', title=None, spacing=10),
        tooltip=[item_col, 'Version', 'Score']
    ).properties(
        height=300,
        title=title
    )
    return chart

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
            # Analyze original
            original_analysis = analyze_text(user_text)
            # Get recommended keywords
            recommended = get_recommended_keywords(original_analysis["seo_keywords"], threshold=0.2)
            # Insert keywords
            updated_text, inserted = insert_keywords(user_text, recommended)
            # Analyze enhanced
            enhanced_analysis = analyze_text(updated_text)


                # --- BEFORE vs AFTER SEO COMPARISON SECTION ---
        st.markdown("---")
        st.markdown("## 🚦 Before vs After SEO Comparison")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📝 Original Text")
            st.markdown(f'<div class="card">{user_text}</div>', unsafe_allow_html=True)
            st.markdown("#### Entities")
            ent_before = [e['id'] for e in original_analysis['entities']]
            st.write(ent_before if ent_before else "None detected")
            st.markdown("#### Topics")
            top_before = [t['label'] for t in original_analysis['topics']]
            st.write(top_before if top_before else "None detected")
            st.markdown("#### Categories")
            cat_before = [c['label'] for c in original_analysis['categories']]
            st.write(cat_before if cat_before else "None detected")
            st.markdown("#### SEO Keywords")
            kw_before = [k['keyword'] for k in original_analysis['seo_keywords']]
            st.write(kw_before if kw_before else "None detected")
        
        with col2:
            st.markdown("### 📝 Enhanced Text")
            st.markdown(f'<div class="card">{updated_text}</div>', unsafe_allow_html=True)
            st.markdown("#### Entities")
            ent_after = [e['id'] for e in enhanced_analysis['entities']]
            st.write(ent_after if ent_after else "None detected")
            st.markdown("#### Topics")
            top_after = [t['label'] for t in enhanced_analysis['topics']]
            st.write(top_after if top_after else "None detected")
            st.markdown("#### Categories")
            cat_after = [c['label'] for c in enhanced_analysis['categories']]
            st.write(cat_after if cat_after else "None detected")
            st.markdown("#### SEO Keywords")
            kw_after = [k['keyword'] for k in enhanced_analysis['seo_keywords']]
            st.write(kw_after if kw_after else "None detected")
        
        # Show a summary table for quick comparison
        st.markdown("### 📋 Quick Comparison Table")
        comp_data = {
            "Original": [
                len(ent_before), len(top_before), len(cat_before), len(kw_before)
            ],
            "Enhanced": [
                len(ent_after), len(top_after), len(cat_after), len(kw_after)
            ]
        }
        comp_df = pd.DataFrame(comp_data, index=["Entities", "Topics", "Categories", "SEO Keywords"])
        st.table(comp_df)
        
        # Show Altair bar charts for visual diff (Entities as example)
        st.markdown("### 📊 Entities Relevance: Before vs After")
        entities_df = compare_items(
            original_analysis["entities"],
            enhanced_analysis["entities"],
            key='id', score='relevance'
        )
        chart = grouped_bar_chart(entities_df, "Entities by Relevance", item_col='Item')
        if chart:
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No entity comparison data available.")
        
        # You can repeat the above for topics, categories, and keywords if you want more charts!
    

        st.markdown("---")
        st.markdown("## 📊 SEO Comparison: Original vs Enhanced")

        # --- Entities Comparison ---
        st.markdown("### 🏷️ Entities Relevance Comparison")
        entities_df = compare_items(
            original_analysis["entities"],
            enhanced_analysis["entities"],
            key='id', score='relevance'
        )
        chart = grouped_bar_chart(entities_df, "Entities by Relevance", item_col='Item')
        if chart:
            st.altair_chart(chart, use_container_width=True)
        st.dataframe(entities_df.sort_values("Enhanced Score", ascending=False), use_container_width=True)

        # --- Topics Comparison ---
        st.markdown("### 🎯 Topics Score Comparison")
        topics_df = compare_items(
            original_analysis["topics"],
            enhanced_analysis["topics"],
            key='label', score='score'
        )
        chart = grouped_bar_chart(topics_df, "Topics by Score", item_col='Item')
        if chart:
            st.altair_chart(chart, use_container_width=True)
        st.dataframe(topics_df.sort_values("Enhanced Score", ascending=False), use_container_width=True)

        # --- Categories Comparison ---
        st.markdown("### 🗂️ Categories Score Comparison")
        categories_df = compare_items(
            original_analysis["categories"],
            enhanced_analysis["categories"],
            key='label', score='score'
        )
        chart = grouped_bar_chart(categories_df, "Categories by Score", item_col='Item')
        if chart:
            st.altair_chart(chart, use_container_width=True)
        st.dataframe(categories_df.sort_values("Enhanced Score", ascending=False), use_container_width=True)

        # --- SEO Keywords Comparison ---
        st.markdown("### 🔑 SEO Keywords Relevance Comparison")
        keywords_df = compare_items(
            original_analysis["seo_keywords"],
            enhanced_analysis["seo_keywords"],
            key='keyword', score='relevance'
        )
        chart = grouped_bar_chart(keywords_df, "SEO Keywords by Relevance", item_col='Item')
        if chart:
            st.altair_chart(chart, use_container_width=True)
        st.dataframe(keywords_df.sort_values("Enhanced Score", ascending=False), use_container_width=True)

        # --- Keyword Presence & Highlights ---
        st.markdown("### 💡 Keyword Presence & Highlights")
        present_before = [kw for kw in recommended if re.search(r'\b' + re.escape(kw.lower()) + r'\b', user_text.lower())]
        present_after = [kw for kw in recommended if re.search(r'\b' + re.escape(kw.lower()) + r'\b', updated_text.lower())]
        added = [kw for kw in present_after if kw not in present_before]

        st.write(f"**Recommended keywords present in original:** {', '.join(present_before) if present_before else 'None'}")
        st.write(f"**Recommended keywords present after enhancement:** {', '.join(present_after) if present_after else 'None'}")
        st.write(f"**New keywords added:** {', '.join(added) if added else 'None'}")

        if added:
            st.markdown("#### 🔍 New Keyword Insertion Snippets")
            snippets = get_keyword_snippets(updated_text, added)
            for snippet in snippets:
                st.markdown(f'<div class="highlight-snippet">…{snippet}…</div>', unsafe_allow_html=True)

        # --- Analytics Insights ---
        st.markdown("### 📈 Analytics Insights")

        def avg_score(df, col):
            if df.empty: return 0
            return df[col].mean()

        insights = []
        # Entities
        orig_entities = len(original_analysis["entities"])
        enh_entities = len(enhanced_analysis["entities"])
        avg_ent_orig = avg_score(entities_df, "Original Score")
        avg_ent_enh = avg_score(entities_df, "Enhanced Score")
        insights.append(f"**Entities:** {orig_entities} → {enh_entities} (avg relevance {avg_ent_orig:.2f} → {avg_ent_enh:.2f})")
        # Topics
        orig_topics = len(original_analysis["topics"])
        enh_topics = len(enhanced_analysis["topics"])
        avg_top_orig = avg_score(topics_df, "Original Score")
        avg_top_enh = avg_score(topics_df, "Enhanced Score")
        insights.append(f"**Topics:** {orig_topics} → {enh_topics} (avg score {avg_top_orig:.2f} → {avg_top_enh:.2f})")
        # Categories
        orig_cats = len(original_analysis["categories"])
        enh_cats = len(enhanced_analysis["categories"])
        avg_cat_orig = avg_score(categories_df, "Original Score")
        avg_cat_enh = avg_score(categories_df, "Enhanced Score")
        insights.append(f"**Categories:** {orig_cats} → {enh_cats} (avg score {avg_cat_orig:.2f} → {avg_cat_enh:.2f})")
        # Keywords
        orig_kw = len(original_analysis["seo_keywords"])
        enh_kw = len(enhanced_analysis["seo_keywords"])
        avg_kw_orig = avg_score(keywords_df, "Original Score")
        avg_kw_enh = avg_score(keywords_df, "Enhanced Score")
        insights.append(f"**SEO Keywords:** {orig_kw} → {enh_kw} (avg relevance {avg_kw_orig:.2f} → {avg_kw_enh:.2f})")

        st.markdown("\n".join(insights))

        # --- Spelling Suggestions ---
        if original_analysis["spelling_suggestions"] or enhanced_analysis["spelling_suggestions"]:
            with st.expander("📝 Spelling Suggestions (Click to expand)"):
                st.write("**Original Text:**")
                for sug in original_analysis["spelling_suggestions"]:
                    st.write(f"**{sug['token']}** ➡ Suggestions: {', '.join(map(str, sug['suggestions']))}")
                st.write("**Enhanced Text:**")
                for sug in enhanced_analysis["spelling_suggestions"]:
                    st.write(f"**{sug['token']}** ➡ Suggestions: {', '.join(map(str, sug['suggestions']))}")

        # --- Show Original vs Enhanced Text ---
        st.markdown("### 📝 Original vs Enhanced Text")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Original Text")
            st.markdown(f'<div class="card">{user_text}</div>', unsafe_allow_html=True)
        with col_b:
            st.markdown("#### Enhanced Text")
            if inserted:
                st.success("Keywords successfully inserted into the text!")
            else:
                st.info("All recommended keywords were already present.")
            st.markdown(f'<div class="card">{updated_text}</div>', unsafe_allow_html=True)

        # --- AI Positive Summary via Groq ---
        st.markdown("---")
        st.markdown("### 🤖 AI SEO Improvement Summary")

        groq_prompt = (
            "Compare the following two texts for SEO optimization. "
            "Highlight only positive improvements and praise the enhanced version, "
            "focusing on new or improved entities, topics, categories, and keywords. "
            "Show keyword highlights and provide a positive meta description. "
            "Do not mention any negative points.\n\n"
            "Original Text:\n"
            f"{user_text}\n\n"
            "Enhanced Text:\n"
            f"{updated_text}\n\n"
            f"Recommended Keywords: {', '.join(recommended)}"
        )
        with st.spinner("Generating AI summary... 🌟"):
            groq_response = groq_ai_request(groq_prompt)

        st.text_area(
            "AI SEO Summary & Meta Description:",
            value=groq_response,
            height=280,
            help="Copy the AI-generated SEO summary and meta description here."
        )

        # --- Copy to Clipboard Button ---
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
        <button class="copy-button" onclick="navigator.clipboard.writeText(document.querySelector('textarea').value)">📋 Copy AI Summary</button>
        """, unsafe_allow_html=True)
