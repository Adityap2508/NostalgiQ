import google.generativeai as genai
import pandas as pd
import re
import json

API_KEY = "AIzaSyA-Sr-VZSTDwqAdgEnx-FLRe8AkaiijurE"  # Paste from step above
genai.configure(api_key=API_KEY)

def extract_text_data(file_path):
    """Extract date and text from a .txt file (format: 'YYYY-MM-DD: text')."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Simple regex: Match date like 2020-07-10: followed by text
        match = re.match(r'(\d{4}-\d{2}-\d{2}):\s*(.*)', line)
        if match:
            date_str, text = match.groups()
            data.append({'date': date_str, 'text': text})
        else:
            # Fallback: Assume whole line is text, no date
            data.append({'date': '2020-01-01', 'text': line})  # Default date
    
    df = pd.DataFrame(data)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df


def extract_image_data(image_path, api_key):
    """Extract date and text from image using Gemini vision."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Upload image
    image_file = genai.upload_file(image_path)
    
    # Prompt Gemini to extract as JSON
    prompt = "Extract all dates and texts from this tweet/convo screenshot (only use the texts sent by the user and not the ones sent by the other person) as a JSON list: [{'date': 'YYYY-MM-DD', 'text': 'post content'}]. Only output JSON."
    response = model.generate_content([prompt, image_file])
    
    # Parse JSON from response
    json_str = response.text.strip('```json\n').strip('\n```')
    data = json.loads(json_str)
    
    df = pd.DataFrame(data)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df


def extract_data(file_path, api_key=None):
    """Unified extractor: Text file or image to DF."""
    if file_path.lower().endswith('.txt'):
        return extract_text_data(file_path)
    elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        if api_key is None:
            raise ValueError("API key required for image extraction")
        return extract_image_data(file_path, api_key)
    else:
        raise ValueError("Unsupported file type—use .txt or image (.png/.jpg).")
    
import torch
from transformers import pipeline
import numpy as np
import gradio as gr
import threading
import time
import queue

# Initialize Whisper model globally
print("Loading Whisper model...")
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base")
print("Whisper model ready!")

def record_audio():
    """
    Simple recording interface - user just records and stops.
    No transcript displayed. Returns transcript as variable in background.
    """
    
    transcript_queue = queue.Queue()
    interface_closed = threading.Event()
    
    def process_recording(audio):
        """Process audio in background and store transcript"""
        if audio is None:
            transcript_queue.put("")
            interface_closed.set()
            return "Recording session ended."
        
        try:
            # Handle Gradio audio format
            if isinstance(audio, tuple):
                sample_rate, audio_data = audio
                
                # Convert to proper format for Whisper
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                
                # Normalize audio data
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                elif audio_data.dtype == np.int32:
                    audio_data = audio_data.astype(np.float32) / 2147483648.0
                
                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    ratio = 16000 / sample_rate
                    new_length = int(len(audio_data) * ratio)
                    audio_data = np.interp(
                        np.linspace(0, len(audio_data), new_length),
                        np.arange(len(audio_data)),
                        audio_data
                    )
                
                # Transcribe in background
                print("🔄 Processing audio...")
                result = transcriber(audio_data)
                transcript = result['text'].strip()
                
                # Store transcript
                transcript_queue.put(transcript)
                interface_closed.set()
                
                return "✅ Recording processed! You can close this window."
            
            else:
                transcript_queue.put("")
                interface_closed.set()
                return "❌ Audio format not supported."
                
        except Exception as e:
            print(f"❌ Background processing error: {str(e)}")
            transcript_queue.put("")
            interface_closed.set()
            return f"❌ Error processing audio."
    
    # Simple recording interface - no transcript shown
    with gr.Blocks(title="Audio Recorder", css=".gradio-container {max-width: 500px}") as interface:
        gr.Markdown("# 🎤 Record Your Audio")
        gr.Markdown("**Instructions:**")
        gr.Markdown("1. Click the microphone button below")
        gr.Markdown("2. Record your audio")
        gr.Markdown("3. Click stop when finished")
        gr.Markdown("4. Your transcript will be processed in the background")
        
        # Audio input - only shows recording interface
        audio_input = gr.Audio(
            sources=["microphone"],
            type="numpy",
            label="🎤 Click to start recording",
            show_label=True
        )
        
        # Hidden textbox for status (minimal)
        status_box = gr.Textbox(
            label="Status",
            value="Ready to record...",
            interactive=False,
            lines=1,
            visible=True
        )
        
        # Auto-process when audio is recorded (no button needed)
        audio_input.change(
            fn=process_recording,
            inputs=audio_input,
            outputs=status_box
        )
    
    print("🎤 Opening recording interface...")
    print("📝 Record your audio - transcript will be processed automatically")
    
    # Launch interface
    interface.launch(
        share=False, 
        inbrowser=True, 
        prevent_thread_lock=True,
        quiet=True
    )
    
    # Wait for processing to complete
    print("⏳ Waiting for recording...")
    
    # Get transcript from queue (blocks until available)
    transcript = transcript_queue.get()
    
    print("✅ Audio processing complete!")
    
    # Automatically close interface after a short delay
    time.sleep(2)
    interface.close()
    
    return transcript


def simple_record():
    """
    Ultra-simple version - just record, no UI feedback at all
    """
    
    result_container = {"transcript": None, "done": False}
    
    def handle_audio(audio):
        if audio is None:
            result_container["transcript"] = ""
            result_container["done"] = True
            return
        
        try:
            if isinstance(audio, tuple):
                sample_rate, audio_data = audio
                
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                
                if sample_rate != 16000:
                    ratio = 16000 / sample_rate
                    new_length = int(len(audio_data) * ratio)
                    audio_data = np.interp(
                        np.linspace(0, len(audio_data), new_length),
                        np.arange(len(audio_data)),
                        audio_data
                    )
                
                print("🔄 Transcribing audio...")
                result = transcriber(audio_data)
                transcript = result['text'].strip()
                
                result_container["transcript"] = transcript
                result_container["done"] = True
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            result_container["transcript"] = ""
            result_container["done"] = True
    
    # Minimal interface - just recording
    interface = gr.Interface(
        fn=lambda audio: handle_audio(audio) or "Recording complete - you can close this window",
        inputs=gr.Audio(
            sources=["microphone"], 
            type="numpy", 
            label="🎤 Record Audio"
        ),
        outputs=gr.Textbox(label="Status", lines=1),
        title="Audio Recorder",
        description="Record your audio. Transcript will be processed automatically.",
        allow_flagging="never"
    )
    
    print("🎤 Simple recorder opening...")
    
    interface.launch(
        share=False, 
        inbrowser=True, 
        prevent_thread_lock=True,
        quiet=True
    )
    
    # Wait for completion
    while not result_container["done"]:
        time.sleep(0.5)
    
    interface.close()
    return result_container["transcript"]


def stealth_record():
    """
    Cleanest version - record and auto-close, transcript returned silently
    """
    
    transcript_result = queue.Queue()
    
    def silent_process(audio):
        """Silently process audio without showing transcript"""
        if audio is None:
            transcript_result.put("")
            return "No recording detected"
        
        try:
            if isinstance(audio, tuple):
                sample_rate, audio_data = audio
                
                # Audio processing
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                
                if sample_rate != 16000:
                    ratio = 16000 / sample_rate
                    new_length = int(len(audio_data) * ratio)
                    audio_data = np.interp(
                        np.linspace(0, len(audio_data), new_length),
                        np.arange(len(audio_data)),
                        audio_data
                    )
                
                # Silent transcription
                result = transcriber(audio_data)
                transcript = result['text'].strip()
                if not transcript: return "Didn't catch that—try again?
                transcript_result.put(transcript)
                
                return "Processing complete ✅"
            
            transcript_result.put("")
            return "Audio format issue"
            
        except Exception as e:
            transcript_result.put("")
            return "Processing error"
    
    # Clean interface
    with gr.Blocks(title="Record", theme=gr.themes.Soft()) as interface:
        gr.Markdown("### 🎤 Record Audio")
        gr.Markdown("Click microphone → Record → Stop recording")
        
        audio_input = gr.Audio(
            sources=["microphone"],
            type="numpy",
            label="",
            show_label=False
        )
        
        # Hidden status (shows only completion)
        status = gr.Textbox(
            value="Ready...",
            label="",
            show_label=False,
            interactive=False,
            lines=1
        )
        
        # Auto-trigger on audio input
        audio_input.change(silent_process, audio_input, status)
    
    interface.launch(share=False, inbrowser=True, prevent_thread_lock=True, quiet=True)
    
    print("🎤 Recording interface ready")
    print("📝 Waiting for your audio...")
    
    # Get result
    transcript = transcript_result.get()
    
    # Auto-close
    threading.Timer(1.0, interface.close).start()
    
    return transcript


# Main usage function
def get_audio_transcript():
    """
    Main function - choose your preferred recording method
    """
    print("🎤 Starting audio recording session...")
    
    # Use the stealth version for cleanest experience
    transcript = stealth_record()
    
    print(f"📝 Transcript captured: '{transcript}'")
    return transcript


# Alternative: Let user choose recording style
def record_with_style(style="stealth"):
    """
    Choose recording style:
    - 'stealth': Clean, minimal UI, auto-close
    - 'simple': Basic interface  
    - 'standard': Standard interface with status
    """
    
    if style == "stealth":
        return stealth_record()
    elif style == "simple":
        return simple_record()
    else:
        return record_audio()
