#!/usr/bin/env python3
"""
Lightweight Personality Prediction Model with Sentiment Analysis

This model predicts personality traits from text using TF-IDF + DistilBERT features
and provides sentiment analysis. Optimized for speed and small datasets.

Installation:
pip install scikit-learn transformers torch numpy pandas nltk

Usage:
python personality_predictor.py
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
from sklearn.pipeline import Pipeline

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Transformers for DistilBERT
from transformers import DistilBertTokenizer, DistilBertModel
import torch

class PersonalityPredictor:
    """
    Lightweight personality prediction model using TF-IDF + DistilBERT features
    """
    
    def __init__(self, max_features_tfidf: int = 5000, max_length: int = 128):
        """
        Initialize the personality predictor
        
        Args:
            max_features_tfidf: Maximum features for TF-IDF vectorization
            max_length: Maximum sequence length for DistilBERT
        """
        self.max_features_tfidf = max_features_tfidf
        self.max_length = max_length
        
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
        self.distilbert_tokenizer = None
        self.distilbert_model = None
        self.personality_classifier = None
        self.sentiment_classifier = None
        self.scaler = StandardScaler()
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize models
        self._initialize_models()
        
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
    
    def _initialize_models(self):
        """Initialize DistilBERT and other models"""
        print("Initializing DistilBERT model...")
        
        # Initialize DistilBERT
        model_name = 'distilbert-base-uncased'
        self.distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.distilbert_model = DistilBertModel.from_pretrained(model_name)
        
        # Set to evaluation mode for faster inference
        self.distilbert_model.eval()
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features_tfidf,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        print("✓ Models initialized successfully!")
    
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
    
    def extract_tfidf_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract TF-IDF features from texts
        
        Args:
            texts: List of text strings
            
        Returns:
            TF-IDF feature matrix
        """
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Fit and transform
        if self.tfidf_vectorizer is None:
            tfidf_features = self.tfidf_vectorizer.fit_transform(processed_texts)
        else:
            tfidf_features = self.tfidf_vectorizer.transform(processed_texts)
        
        return tfidf_features.toarray()
    
    def extract_distilbert_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract DistilBERT embeddings from texts
        
        Args:
            texts: List of text strings
            
        Returns:
            DistilBERT feature matrix
        """
        features = []
        
        with torch.no_grad():
            for text in texts:
                # Preprocess text
                processed_text = self.preprocess_text(text)
                
                # Tokenize
                inputs = self.distilbert_tokenizer(
                    processed_text,
                    return_tensors='pt',
                    max_length=self.max_length,
                    padding=True,
                    truncation=True
                )
                
                # Get embeddings
                outputs = self.distilbert_model(**inputs)
                
                # Use [CLS] token embedding (first token)
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
                features.append(embedding.flatten())
        
        return np.array(features)
    
    def extract_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract combined TF-IDF + DistilBERT features
        
        Args:
            texts: List of text strings
            
        Returns:
            Combined feature matrix
        """
        print("Extracting TF-IDF features...")
        tfidf_features = self.extract_tfidf_features(texts)
        
        print("Extracting DistilBERT features...")
        distilbert_features = self.extract_distilbert_features(texts)
        
        # Combine features
        combined_features = np.hstack([tfidf_features, distilbert_features])
        
        print(f"✓ Feature extraction complete. Shape: {combined_features.shape}")
        return combined_features
    
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
            
            # Pessimistic/Negative
            "Nothing ever goes right for me. I'm doomed to fail.",
            "The world is a terrible place and people are awful.",
            "Why even try when everything always ends badly?",
            
            # Friendly/Sociable
            "Hey everyone! How's your day going? Let's hang out soon!",
            "I love meeting new people and making friends wherever I go.",
            "Thanks so much for your help! You're such a wonderful person!",
            
            # Sarcastic/Witty
            "Oh great, another Monday. Just what I needed to start my week.",
            "Sure, let's have another meeting about having meetings. Brilliant.",
            "I'm not saying you're wrong, but you're not right either.",
            
            # Formal/Polite
            "I would be most grateful if you could kindly assist me with this matter.",
            "Thank you for your time and consideration. I look forward to your response.",
            "I respectfully request that you review the attached documentation.",
            
            # Informal/Casual
            "yo what's up? wanna grab some food later?",
            "lol that's hilarious! can't stop laughing",
            "nah i'm good, thanks tho! catch you later",
            
            # Creative/Imaginative
            "I had this amazing dream where I was flying through clouds of cotton candy.",
            "What if we could paint with colors that don't exist yet?",
            "I love creating stories about magical creatures in hidden worlds.",
            
            # Logical/Analytical
            "Based on the data analysis, we can conclude that the hypothesis is supported.",
            "Let me break this down step by step to understand the problem better.",
            "The correlation coefficient suggests a strong positive relationship.",
            
            # Emotional/Sensitive
            "I feel so overwhelmed by all these emotions right now.",
            "That movie made me cry so much, it touched my heart deeply.",
            "I'm really sensitive to other people's feelings and energy.",
            
            # Confident/Assertive
            "I know exactly what I want and I'm going to get it.",
            "I'm the best person for this job and I can prove it.",
            "I don't need anyone's approval to pursue my dreams.",
            
            # Introverted/Reflective
            "I need some quiet time alone to think and recharge.",
            "I prefer deep conversations with close friends over large parties.",
            "I often reflect on life and enjoy my own company.",
            
            # Extroverted/Outgoing
            "I love being around people and get energy from social interactions!",
            "Let's go to that party! I want to meet everyone there!",
            "I'm always up for new adventures with friends!",
            
            # Curious/Inquisitive
            "I wonder how that works? Let me research it more.",
            "Why do things happen the way they do? I need to understand.",
            "I'm always asking questions and learning new things.",
            
            # Humorous/Playful
            "Why don't scientists trust atoms? Because they make up everything!",
            "I told my wife she was drawing her eyebrows too high. She looked surprised.",
            "Life is too short to be serious all the time! Let's have fun!",
            
            # Motivational/Inspirational
            "You can achieve anything you set your mind to! Believe in yourself!",
            "Every expert was once a beginner. Keep pushing forward!",
            "Your potential is limitless. Don't let fear hold you back!"
        ]
        
        # Create personality labels (one-hot encoded)
        personality_labels = [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # optimistic
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # optimistic
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # optimistic
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # pessimistic
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # pessimistic
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # pessimistic
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # friendly
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # friendly
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # friendly
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # sarcastic
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # sarcastic
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # sarcastic
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # formal
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # formal
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # formal
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # informal
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # informal
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # informal
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # creative
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # creative
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # creative
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # logical
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # logical
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # logical
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # emotional
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # emotional
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # emotional
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # confident
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # confident
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # confident
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # introverted
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # introverted
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # introverted
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # extroverted
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # extroverted
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # extroverted
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # curious
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # curious
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # curious
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # humorous
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # humorous
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # humorous
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # motivational
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # motivational
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # motivational
        ]
        
        # Create sentiment labels
        sentiment_labels = [
            'positive', 'positive', 'positive',  # optimistic
            'negative', 'negative', 'negative',  # pessimistic
            'positive', 'positive', 'positive',  # friendly
            'negative', 'negative', 'neutral',   # sarcastic
            'neutral', 'neutral', 'neutral',     # formal
            'neutral', 'positive', 'neutral',    # informal
            'positive', 'positive', 'positive',  # creative
            'neutral', 'neutral', 'neutral',     # logical
            'negative', 'negative', 'neutral',   # emotional
            'positive', 'positive', 'positive',  # confident
            'neutral', 'neutral', 'neutral',     # introverted
            'positive', 'positive', 'positive',  # extroverted
            'neutral', 'neutral', 'neutral',     # curious
            'positive', 'positive', 'positive',  # humorous
            'positive', 'positive', 'positive',  # motivational
        ]
        
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
        
        # Extract features
        features = self.extract_features(texts)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Split data
        X_train, X_test, y_personality_train, y_personality_test, y_sentiment_train, y_sentiment_test = train_test_split(
            features_scaled, personality_labels, sentiment_labels, 
            test_size=0.2, random_state=42
        )
        
        # Train personality classifier (multi-label)
        print("Training personality classifier...")
        self.personality_classifier = LogisticRegression(
            random_state=42, 
            max_iter=1000,
            class_weight='balanced'
        )
        
        # Convert personality labels to numpy array
        y_personality_train = np.array(y_personality_train)
        y_personality_test = np.array(y_personality_test)
        
        # Train on each personality trait separately
        personality_predictions = []
        for i, trait in enumerate(self.personality_traits):
            print(f"  Training {trait}...")
            self.personality_classifier.fit(X_train, y_personality_train[:, i])
            pred = self.personality_classifier.predict(X_test)
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
        if self.personality_classifier is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Extract features
        features = self.extract_features(texts)
        features_scaled = self.scaler.transform(features)
        
        # Predict personality traits
        personality_scores = []
        for i, trait in enumerate(self.personality_traits):
            # Get probability scores
            proba = self.personality_classifier.predict_proba(features_scaled)
            if proba.shape[1] == 2:  # Binary classification
                scores = proba[:, 1]  # Probability of positive class
            else:
                scores = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
            personality_scores.append(scores)
        
        # Convert to list of dictionaries
        results = []
        for i in range(len(texts)):
            trait_scores = {}
            for j, trait in enumerate(self.personality_traits):
                trait_scores[trait] = float(personality_scores[j][i])
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
        
        # Extract features
        features = self.extract_features(texts)
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
    """Example usage of the PersonalityPredictor"""
    print("=== Lightweight Personality Prediction Model ===\n")
    
    # Initialize predictor
    predictor = PersonalityPredictor()
    
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
