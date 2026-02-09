#!/usr/bin/env python3
"""
Demo script for Talking Video Generator

This script demonstrates the basic usage of the talking video generator
with a simple example.
"""

import os
import cv2
import numpy as np
from pathlib import Path

def create_demo_image():
    """Create a demo face image for testing"""
    print("🖼️ Creating demo face image...")
    
    # Create a simple face image
    img = np.ones((512, 512, 3), dtype=np.uint8) * 255
    
    # Draw a more realistic face
    # Face outline
    cv2.ellipse(img, (256, 256), (120, 150), 0, 0, 360, (220, 200, 180), -1)
    cv2.ellipse(img, (256, 256), (120, 150), 0, 0, 360, (180, 160, 140), 2)
    
    # Eyes
    cv2.circle(img, (220, 220), 15, (0, 0, 0), -1)
    cv2.circle(img, (292, 220), 15, (0, 0, 0), -1)
    cv2.circle(img, (220, 220), 8, (255, 255, 255), -1)
    cv2.circle(img, (292, 220), 8, (255, 255, 255), -1)
    
    # Eyebrows
    cv2.ellipse(img, (220, 200), (20, 5), 0, 0, 180, (0, 0, 0), 2)
    cv2.ellipse(img, (292, 200), (20, 5), 0, 0, 180, (0, 0, 0), 2)
    
    # Nose
    cv2.ellipse(img, (256, 280), (8, 15), 0, 0, 360, (200, 180, 160), -1)
    
    # Mouth
    cv2.ellipse(img, (256, 320), (25, 12), 0, 0, 180, (0, 0, 0), 3)
    
    # Add some texture
    for i in range(100):
        x = np.random.randint(150, 362)
        y = np.random.randint(150, 362)
        cv2.circle(img, (x, y), 1, (200, 180, 160), -1)
    
    # Save the image
    cv2.imwrite("demo_face.jpg", img)
    print("✅ Demo face image created: demo_face.jpg")
    
    return "demo_face.jpg"

def run_demo():
    """Run the talking video generator demo"""
    print("🎬 Talking Video Generator Demo")
    print("=" * 40)
    
    # Check if SadTalker is set up
    if not os.path.exists("SadTalker"):
        print("❌ SadTalker not found. Please run setup first:")
        print("python setup_talking_video.py")
        return
    
    # Create demo image
    demo_image = create_demo_image()
    
    # Demo text
    demo_text = "Hello! I'm a demo talking head. This is a demonstration of the talking video generator using SadTalker and Coqui TTS."
    
    print(f"\n📝 Demo text: {demo_text}")
    print(f"🖼️ Demo image: {demo_image}")
    
    # Import and use the generator
    try:
        from talking_video_generator import TalkingVideoGenerator
        
        print("\n🤖 Initializing generator...")
        generator = TalkingVideoGenerator()
        
        print("\n🎬 Generating talking video...")
        result = generator.process_single(
            input_image=demo_image,
            text=demo_text,
            output_name="demo_video"
        )
        
        if result:
            print(f"\n🎉 Demo video generated successfully!")
            print(f"📁 Output: {result}")
            print(f"📂 Location: {os.path.abspath(result)}")
        else:
            print("\n❌ Demo video generation failed")
            
    except ImportError as e:
        print(f"❌ Error importing generator: {e}")
        print("Make sure all dependencies are installed.")
    except Exception as e:
        print(f"❌ Error during generation: {e}")

def show_usage_examples():
    """Show usage examples"""
    print("\n📚 Usage Examples:")
    print("=" * 30)
    
    examples = [
        {
            "title": "Single Video Generation",
            "command": "python talking_video_generator.py --image person.jpg --text 'Hello world!'",
            "description": "Generate a single talking video from an image and text"
        },
        {
            "title": "Custom Output Name",
            "command": "python talking_video_generator.py --image person.jpg --text 'Hello!' --output my_video",
            "description": "Generate video with custom output name"
        },
        {
            "title": "Different TTS Model",
            "command": "python talking_video_generator.py --image person.jpg --text 'Hello!' --tts-model 'tts_models/en/vctk/vits'",
            "description": "Use a different TTS model for different voice"
        },
        {
            "title": "Batch Processing",
            "command": "python talking_video_generator.py --batch example_batch.json",
            "description": "Process multiple videos from a JSON file"
        },
        {
            "title": "Interactive Mode",
            "command": "python talking_video_generator.py",
            "description": "Run in interactive mode with prompts"
        },
        {
            "title": "List TTS Models",
            "command": "python talking_video_generator.py --list-models",
            "description": "Show all available TTS models"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}")
        print(f"   Command: {example['command']}")
        print(f"   Description: {example['description']}")

def main():
    """Main demo function"""
    print("🎬 Talking Video Generator - Demo & Examples")
    print("=" * 50)
    
    # Show usage examples
    show_usage_examples()
    
    # Ask if user wants to run demo
    print("\n" + "=" * 50)
    response = input("Would you like to run the demo? (y/n): ").strip().lower()
    
    if response in ['y', 'yes']:
        run_demo()
    else:
        print("Demo skipped. You can run it later with: python demo_talking_video.py")
    
    print("\n🎉 Demo complete! Check the README for more information.")

if __name__ == "__main__":
    main()
