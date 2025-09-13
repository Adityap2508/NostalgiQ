#!/usr/bin/env python3
"""
Lightweight Personality Prediction Model (Simplified Version)

This version uses only TF-IDF features for faster execution and minimal dependencies.
Perfect for quick testing and small datasets.

Installation:
pip install scikit-learn numpy pandas nltk

Usage:
python personality_predictor_simple.py
"""

import numpy as np
import pandas as pd
import re
import string
from typing import List, Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class SimplePersonalityPredictor:
    """
    Lightweight personality prediction model using only TF-IDF features
    """
    
    def __init__(self, max_features: int = 3000):
        """
        Initialize the personality predictor
        
        Args:
            max_features: Maximum features for TF-IDF vectorization
        """
        self.max_features = max_features
        
        # Personality traits (15 categories)
        self.personality_traits = [
            'optimistic_positive',
            'pessimistic_negative', 
            'friendly_sociable',
            'sarcastic_witty',
            'formal_polite',
            'informal_casual',
            'creative_imaginative',
            'logical_analytical',
            'emotional_sensitive',
            'confident_assertive',
            'introverted_reflective',
            'extroverted_outgoing',
            'curious_inquisitive',
            'humorous_playful',
            'motivational_inspirational'
        ]
        
        # Sentiment categories
        self.sentiment_labels = ['negative', 'neutral', 'positive']
        
        # Initialize components
        self.tfidf_vectorizer = None
        self.personality_classifiers = {}
        self.sentiment_classifier = None
        self.scaler = StandardScaler()
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            lowercase=True
        )
        
        print("✓ Simple Personality Predictor initialized!")
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text: lowercase, remove punctuation, filter stopwords
        
        Args:
            text: Input text string
            
        Returns:
            Preprocessed text string
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove punctuation but keep basic sentence structure
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        return text
    
    def create_sample_data(self) -> Tuple[List[str], List[List[float]], List[str]]:
        """
        Create sample training data with personality traits and sentiment
        
        Returns:
            Tuple of (texts, personality_labels, sentiment_labels)
        """
        # Sample texts with different personality traits
        sample_texts = [
            # Optimistic/Positive
            "I'm so excited about the future! Everything is going to be amazing!",
            "Life is beautiful and full of opportunities waiting to be discovered.",
            "Every challenge is just a stepping stone to success!",
            "I believe in the power of positive thinking and hard work.",
            "The best is yet to come and I can't wait to see what happens!",
            
            # Pessimistic/Negative
            "Nothing ever goes right for me. I'm doomed to fail.",
            "The world is a terrible place and people are awful.",
            "Why even try when everything always ends badly?",
            "I'm so tired of being disappointed by everything.",
            "Life is just one disappointment after another.",
            
            # Friendly/Sociable
            "Hey everyone! How's your day going? Let's hang out soon!",
            "I love meeting new people and making friends wherever I go.",
            "Thanks so much for your help! You're such a wonderful person!",
            "Let's all work together and make this a great experience!",
            "I'm so grateful for all the amazing people in my life!",
            
            # Sarcastic/Witty
            "Oh great, another Monday. Just what I needed to start my week.",
            "Sure, let's have another meeting about having meetings. Brilliant.",
            "I'm not saying you're wrong, but you're not right either.",
            "Well, that went about as well as expected.",
            "Nothing says success like a perfectly planned disaster.",
            
            # Formal/Polite
            "I would be most grateful if you could kindly assist me with this matter.",
            "Thank you for your time and consideration. I look forward to your response.",
            "I respectfully request that you review the attached documentation.",
            "I would appreciate the opportunity to discuss this further.",
            "Please accept my sincere gratitude for your assistance.",
            
            # Informal/Casual
            "yo what's up? wanna grab some food later?",
            "lol that's hilarious! can't stop laughing",
            "nah i'm good, thanks tho! catch you later",
            "omg that's so cool! i love it!",
            "hey dude, how's it going?",
            
            # Creative/Imaginative
            "I had this amazing dream where I was flying through clouds of cotton candy.",
            "What if we could paint with colors that don't exist yet?",
            "I love creating stories about magical creatures in hidden worlds.",
            "Imagine a world where music could be tasted and colors could be heard.",
            "I see art in everything around me, even in the most ordinary moments.",
            
            # Logical/Analytical
            "Based on the data analysis, we can conclude that the hypothesis is supported.",
            "Let me break this down step by step to understand the problem better.",
            "The correlation coefficient suggests a strong positive relationship.",
            "We need to examine the evidence systematically before drawing conclusions.",
            "The statistical significance of these results cannot be ignored.",
            
            # Emotional/Sensitive
            "I feel so overwhelmed by all these emotions right now.",
            "That movie made me cry so much, it touched my heart deeply.",
            "I'm really sensitive to other people's feelings and energy.",
            "Sometimes I feel like I absorb everyone else's emotions.",
            "I can't help but feel deeply affected by the world around me.",
            
            # Confident/Assertive
            "I know exactly what I want and I'm going to get it.",
            "I'm the best person for this job and I can prove it.",
            "I don't need anyone's approval to pursue my dreams.",
            "I'm confident in my abilities and I know I can succeed.",
            "I believe in myself and I'm not afraid to take risks.",
            
            # Introverted/Reflective
            "I need some quiet time alone to think and recharge.",
            "I prefer deep conversations with close friends over large parties.",
            "I often reflect on life and enjoy my own company.",
            "I find peace in solitude and quiet contemplation.",
            "I'm most comfortable in small, intimate settings.",
            
            # Extroverted/Outgoing
            "I love being around people and get energy from social interactions!",
            "Let's go to that party! I want to meet everyone there!",
            "I'm always up for new adventures with friends!",
            "The more people, the better! I thrive in crowds!",
            "I can't wait to network and meet new people at this event!",
            
            # Curious/Inquisitive
            "I wonder how that works? Let me research it more.",
            "Why do things happen the way they do? I need to understand.",
            "I'm always asking questions and learning new things.",
            "There's so much to discover and explore in this world.",
            "I love diving deep into topics that interest me.",
            
            # Humorous/Playful
            "Why don't scientists trust atoms? Because they make up everything!",
            "I told my wife she was drawing her eyebrows too high. She looked surprised.",
            "Life is too short to be serious all the time! Let's have fun!",
            "I love making people laugh and bringing joy to others.",
            "Humor is the best medicine for any situation!",
            
            # Motivational/Inspirational
            "You can achieve anything you set your mind to! Believe in yourself!",
            "Every expert was once a beginner. Keep pushing forward!",
            "Your potential is limitless. Don't let fear hold you back!",
            "Success is not final, failure is not fatal. Keep going!",
            "The only way to do great work is to love what you do!",
        ]
        
        # Create personality labels (one-hot encoded)
        personality_labels = []
        for i in range(0, len(sample_texts), 5):
            trait_index = i // 5
            for j in range(5):
                label = [0.0] * 15
                if trait_index < 15:
                    label[trait_index] = 1.0
                personality_labels.append(label)
        
        # Create sentiment labels
        sentiment_labels = []
        for i in range(0, len(sample_texts), 5):
            trait_index = i // 5
            if trait_index in [0, 2, 4, 6, 9, 11, 13, 14]:  # positive traits
                sentiment = ['positive'] * 5
            elif trait_index in [1, 3, 8]:  # negative traits
                sentiment = ['negative'] * 5
            else:  # neutral traits
                sentiment = ['neutral'] * 5
            sentiment_labels.extend(sentiment)
        
        return sample_texts, personality_labels, sentiment_labels
    
    def train(self, texts: List[str] = None, personality_labels: List[List[float]] = None, 
              sentiment_labels: List[str] = None):
        """
        Train the personality and sentiment classifiers
        
        Args:
            texts: List of training texts
            personality_labels: List of personality trait vectors
            sentiment_labels: List of sentiment labels
        """
        # Use sample data if none provided
        if texts is None:
            texts, personality_labels, sentiment_labels = self.create_sample_data()
        
        print(f"Training on {len(texts)} samples...")
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Extract TF-IDF features
        print("Extracting TF-IDF features...")
        tfidf_features = self.tfidf_vectorizer.fit_transform(processed_texts)
        features = tfidf_features.toarray()
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Split data
        X_train, X_test, y_personality_train, y_personality_test, y_sentiment_train, y_sentiment_test = train_test_split(
            features_scaled, personality_labels, sentiment_labels, 
            test_size=0.2, random_state=42
        )
        
        # Train personality classifiers (one for each trait)
        print("Training personality classifiers...")
        y_personality_train = np.array(y_personality_train)
        y_personality_test = np.array(y_personality_test)
        
        personality_predictions = []
        for i, trait in enumerate(self.personality_traits):
            print(f"  Training {trait}...")
            classifier = LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight='balanced'
            )
            classifier.fit(X_train, y_personality_train[:, i])
            self.personality_classifiers[trait] = classifier
            
            # Predict on test set
            pred = classifier.predict(X_test)
            personality_predictions.append(pred)
        
        # Train sentiment classifier
        print("Training sentiment classifier...")
        self.sentiment_classifier = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        self.sentiment_classifier.fit(X_train, y_sentiment_train)
        
        # Evaluate
        print("\n=== EVALUATION RESULTS ===")
        
        # Personality evaluation
        personality_pred = np.array(personality_predictions).T
        print("Personality Traits Accuracy:")
        for i, trait in enumerate(self.personality_traits):
            acc = accuracy_score(y_personality_test[:, i], personality_pred[:, i])
            print(f"  {trait}: {acc:.3f}")
        
        # Sentiment evaluation
        sentiment_pred = self.sentiment_classifier.predict(X_test)
        sentiment_acc = accuracy_score(y_sentiment_test, sentiment_pred)
        print(f"\nSentiment Accuracy: {sentiment_acc:.3f}")
        
        print("\n✓ Training completed successfully!")
    
    def predict_personality(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Predict personality traits for given texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of dictionaries with personality trait scores
        """
        if not self.personality_classifiers:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Extract features
        features = self.tfidf_vectorizer.transform(processed_texts).toarray()
        features_scaled = self.scaler.transform(features)
        
        # Predict personality traits
        results = []
        for i in range(len(texts)):
            trait_scores = {}
            for trait, classifier in self.personality_classifiers.items():
                # Get probability scores
                proba = classifier.predict_proba(features_scaled[i:i+1])
                if proba.shape[1] == 2:  # Binary classification
                    score = proba[0, 1]  # Probability of positive class
                else:
                    score = proba[0, 1] if proba.shape[1] > 1 else proba[0, 0]
                trait_scores[trait] = float(score)
            results.append(trait_scores)
        
        return results
    
    def predict_sentiment(self, texts: List[str]) -> List[str]:
        """
        Predict sentiment for given texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of sentiment labels
        """
        if self.sentiment_classifier is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Extract features
        features = self.tfidf_vectorizer.transform(processed_texts).toarray()
        features_scaled = self.scaler.transform(features)
        
        # Predict sentiment
        predictions = self.sentiment_classifier.predict(features_scaled)
        
        return predictions.tolist()
    
    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict both personality traits and sentiment for given texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of dictionaries with personality and sentiment predictions
        """
        personality_scores = self.predict_personality(texts)
        sentiment_predictions = self.predict_sentiment(texts)
        
        results = []
        for i, text in enumerate(texts):
            result = {
                'text': text,
                'personality_traits': personality_scores[i],
                'sentiment': sentiment_predictions[i],
                'top_traits': self._get_top_traits(personality_scores[i], top_k=3)
            }
            results.append(result)
        
        return results
    
    def _get_top_traits(self, trait_scores: Dict[str, float], top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Get top-k personality traits by score
        
        Args:
            trait_scores: Dictionary of trait scores
            top_k: Number of top traits to return
            
        Returns:
            List of (trait_name, score) tuples
        """
        sorted_traits = sorted(trait_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_traits[:top_k]

def main():
    """Example usage of the SimplePersonalityPredictor"""
    print("=== Lightweight Personality Prediction Model (Simple Version) ===\n")
    
    # Initialize predictor
    predictor = SimplePersonalityPredictor()
    
    # Train the model
    predictor.train()
    
    # Example texts for prediction
    test_texts = [
        "I'm so excited about this new project! It's going to be amazing!",
        "I don't think this will work out. Nothing ever goes right for me.",
        "Hey everyone! Let's all work together and make this happen!",
        "Oh great, another meeting. Just what I needed today.",
        "I would appreciate it if you could review the attached document.",
        "yo what's up? wanna grab lunch later?",
        "I had this incredible dream about flying through rainbow clouds!",
        "Based on the statistical analysis, we can conclude that...",
        "I feel so overwhelmed by all these emotions right now.",
        "I know I'm the best person for this job and I can prove it!",
        "I need some quiet time alone to think and recharge.",
        "Let's go to that party! I want to meet everyone there!",
        "I wonder how that works? Let me research it more.",
        "Why don't scientists trust atoms? Because they make up everything!",
        "You can achieve anything you set your mind to! Believe in yourself!"
    ]
    
    print("\n=== PREDICTION RESULTS ===")
    
    # Make predictions
    results = predictor.predict(test_texts)
    
    # Display results
    for i, result in enumerate(results):
        print(f"\nText {i+1}: {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        print("Top Personality Traits:")
        for trait, score in result['top_traits']:
            print(f"  {trait}: {score:.3f}")
        print("-" * 50)

if __name__ == "__main__":
    main()
