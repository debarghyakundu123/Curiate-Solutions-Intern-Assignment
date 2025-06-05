import os
import re
import streamlit as st
import textrazor
from groq import Groq

# --- Setup ---

# TextRazor API key
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

# Groq API key (or environment variable)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or "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"
groq_client = Groq(api_key=GROQ_API_KEY)

# --- Functions ---

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
    relations = []
    categories = []
    words = []
    spelling_suggestions = []

    # Extract entities
    for entity in response.entities():
        entities.append({
            "id": entity.id,
            "relevance": entity.relevance_score,
            "confidence": entity.confidence_score,
        })
        seo_keywords.append({"keyword": entity.id, "relevance": entity.relevance_score})

    # Extract topics
    for topic in response.topics():
        topics.append({"label": topic.label, "score": topic.score})

    # Extract relations
    for rel in response.relations():
        relations.append({"id": rel.id})

    # Extract categories
    for cat in response.categories():
        categories.append({"label": cat.label, "score": cat.score})

    # Extract words and spelling suggestions
    for word in response.words():
        words.append({
            "token": word.token,
            "stem": word.stem,
            "lemma": word.lemma,
            "spelling_suggestions": word.spelling_suggestions
        })
        if word.spelling_suggestions:
            # spelling_suggestions is a list of strings normally
            spelling_suggestions.append({
                "token": word.token,
                "suggestions": word.spelling_suggestions
            })

    return {
        "seo_keywords": seo_keywords,
        "entities": entities,
        "topics": topics,
        "relations": relations,
        "categories": categories,
        "words": words,
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

# --- Streamlit App ---

st.set_page_config(page_title="SEO Text Analyzer & AI Enhancer", layout="wide")

st.title("SEO Text Analyzer & AI Enhancer")
st.markdown(
    "Enter your text below to analyze it using TextRazor, "
    "extract SEO-relevant keywords and metrics, "
    "and get AI-generated SEO improvement suggestions."
)

user_text = st.text_area("Enter text to analyze:", height=200)

if st.button("Analyze Text"):
    if not user_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text with TextRazor..."):
            analysis = analyze_text(user_text)

        # Display TextRazor analysis metrics
        st.header("TextRazor Analysis Results")

        # Entities table
        if analysis["entities"]:
            st.subheader("Entities")
            st.table([
                {
                    "Entity": e["id"],
                    "Relevance": f"{e['relevance']:.4f}",
                    "Confidence": f"{e['confidence']:.4f}"
                }
                for e in analysis["entities"]
            ])
        else:
            st.info("No entities detected.")

        # Topics
        if analysis["topics"]:
            st.subheader("Topics")
            st.table([
                {"Topic": t["label"], "Score": f"{t['score']:.4f}"}
                for t in analysis["topics"]
            ])

        # Relations
        if analysis["relations"]:
            st.subheader("Relations")
            # Only include relations that have an 'id' key
            relation_ids = [r["id"] for r in analysis["relations"] if isinstance(r, dict) and "id" in r]
            if relation_ids:
                st.write(", ".join(relation_ids))
            else:
                st.info("No valid relations with 'id' found.")

        # Categories
        if analysis["categories"]:
            st.subheader("Categories")
            st.table([
                {"Category": c["label"], "Score": f"{c['score']:.4f}"}
                for c in analysis["categories"]
            ])

        # Spelling suggestions
        if analysis["spelling_suggestions"]:
            st.subheader("Spelling Suggestions")
            for sug in analysis["spelling_suggestions"]:
                # Join suggestions safely assuming list of strings
                st.write(f"**{sug['token']}** ➡ Suggestions: {', '.join(map(str, sug['suggestions']))}")

        # SEO Keywords and Recommendations
        seo_keywords = analysis["seo_keywords"]

        st.header("SEO Keywords and Recommendations")
        if seo_keywords:
            seo_keywords_sorted = sorted(seo_keywords, key=lambda x: x["relevance"], reverse=True)
            st.table([
                {"Keyword": kw["keyword"], "Relevance": f"{kw['relevance']:.4f}"}
                for kw in seo_keywords_sorted
            ])
        else:
            st.info("No SEO keywords detected.")

        recommended = get_recommended_keywords(seo_keywords, threshold=0.2)

        if recommended:
            st.subheader("Recommended Keywords (Relevance >= 0.2)")
            st.write(", ".join(recommended))
        else:
            st.info("No recommended keywords found based on threshold.")

        # Insert recommended keywords into original text
        updated_text, inserted = insert_keywords(user_text, recommended)
        st.header("Text After Inserting Recommended Keywords")
        if inserted:
            st.success("Recommended keywords inserted into text:")
        else:
            st.info("All recommended keywords already present in the text.")
        st.write(updated_text)

        # Prepare prompt for AI
        groq_prompt = (
            f"Analyze the following text for SEO optimization and explain how to improve it:\n\n"
            f"{updated_text}\n\n"
            f"Also, suggest a meta description based on the text and recommended keywords: {', '.join(recommended)}"
        )

        st.header("AI SEO Improvement Suggestions (Groq)")

        with st.spinner("Getting AI suggestions..."):
            groq_response = groq_ai_request(groq_prompt)

        st.text_area("AI Suggestions and Meta Description:", value=groq_response, height=250)
