#!/usr/bin/env python3
"""
Quick test of the personality prediction model
"""

def test_imports():
    """Test if all required packages can be imported"""
    try:
        import sklearn
        print("✓ scikit-learn available")
        
        import transformers
        print("✓ transformers available")
        
        import torch
        print("✓ torch available")
        
        import numpy as np
        print("✓ numpy available")
        
        import pandas as pd
        print("✓ pandas available")
        
        import nltk
        print("✓ nltk available")
        
        print("\n✓ All packages imported successfully!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_model_creation():
    """Test if the model can be created"""
    try:
        from personality_predictor import PersonalityPredictor
        
        print("Creating PersonalityPredictor...")
        predictor = PersonalityPredictor(max_features_tfidf=1000, max_length=64)
        print("✓ Model created successfully!")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing Personality Prediction Model ===\n")
    
    # Test imports
    if not test_imports():
        print("Please install missing packages:")
        print("pip install scikit-learn transformers torch numpy pandas nltk")
        exit(1)
    
    # Test model creation
    if not test_model_creation():
        print("Model creation failed. Check the error above.")
        exit(1)
    
    print("\n✓ All tests passed! The model is ready to use.")
    print("\nTo run the full model:")
    print("python personality_predictor.py")
