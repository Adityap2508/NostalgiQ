import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import google.generativeai as genai

# Load sentiment model once (global, so it's ready)
sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

def create_snapshots(df):
    """Creates personality snapshots by year with top words and sentiment."""
    df = df.copy()
    df['year'] = df['date'].dt.year
    grouped = df.groupby('year')
    
    snapshots = {}
    for year, group in grouped:
        texts = group['text'].tolist()
        if len(texts) == 0:
            continue
        
        # TF-IDF top words
        vectorizer = TfidfVectorizer(max_features=4, stop_words='english')
        tfidf = vectorizer.fit_transform(texts)
        top_words = vectorizer.get_feature_names_out().tolist()
        
        # Sentiment analysis
        sentiments = sentiment_analyzer(texts)
        labels = [s['label'] for s in sentiments]
        avg_sentiment = max(set(labels), key=labels.count)
        avg_score = sum(s['score'] for s in sentiments) / len(sentiments)
        
        snapshots[year] = {
            'texts': texts,
            'top_words': top_words,
            'sentiment': f"{avg_sentiment} ({avg_score:.2f})"
        }
    
    return snapshots

API_KEY = "AIzaSyA-Sr-VZSTDwqAdgEnx-FLRe8AkaiijurE"  # Paste your key here
genai.configure(api_key=API_KEY)

# Load model with short config (global)
model = genai.GenerativeModel(
    'gemini-1.5-flash',
    generation_config=genai.types.GenerationConfig(
        max_output_tokens=80,
        temperature=0.7
    )
)

def generate_response(snap, user_message):
    """Generate a short response as past self from snapshot."""
    year = 2019  # Hardcode for now; app will pass it
    style_desc = f"You are me from {year}. Vibe: {snap['sentiment']}, top interests: {', '.join(snap['top_words'])}. Examples from my posts: {' '.join(snap['texts'])}."
    prompt = f"{style_desc}\n\nTopic: {user_message}\nYour response as past me: Keep your response to 1-2 sentences only."
    
    response = model.generate_content(prompt)
    return response.text.strip()

    # Things to improve in the model - More words in tfidf and other enhancements in sentiment analysis, allowing gemini to learn from its responses (reinforcement learning)