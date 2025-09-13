# Windows Setup Instructions

## If you encounter setuptools errors:

1. **Update pip and setuptools:**
   ```bash
   python -m pip install --upgrade pip
   pip install --upgrade setuptools wheel
   ```

2. **Install packages individually:**
   ```bash
   pip install numpy
   pip install pillow
   pip install opencv-python
   pip install gtts
   pip install torch torchvision torchaudio
   ```

3. **For SadTalker models (if download fails):**
   - Visit: https://github.com/OpenTalker/SadTalker
   - Download models manually to SadTalker/checkpoints/
   - Or use the simplified version without SadTalker

## Alternative: Use the Simple Version

If SadTalker setup continues to fail, use the simple version:
```bash
python talking_video_simple.py --image face.jpg --text "Hello!"
```

This version only requires:
- Google TTS (gtts)
- Basic image processing
- No SadTalker dependency
