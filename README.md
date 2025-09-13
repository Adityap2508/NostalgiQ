
# Talking Video & Personality Predictor Suite

This repository bundles small utilities and example scripts to generate talking head videos from a single image, run personality prediction, and related helper scripts. It collects several approaches (SadTalker, Wav2Lip, etc.) and convenience scripts to download models and run demos. The project appears to be assembled for demos and research experiments rather than a production-ready product.

Files of interest
- `talking_video_generator.py`, `talking_video_simple.py`, `talking_video.py`, `simple_video.py`, `working_talking_video.py` — main scripts to create talking videos from images and audio.
- `sadtalker_working.py`, `SadTalker/` — SadTalker integration.
- `download_sadtalker_models.py`, `download_sadtalker_direct.py`, `setup_sadtalker_manual.py` — helpers to fetch and prepare SadTalker models.
- `setup_wav2lip.py`, `setup_talking_video.py`, `setup_simple.py`, `setup_windows.py`, `setup_sadtalker_manual.py` — environment / installation helpers.
- `personality_predictor.py`, `personality_predictor_simple.py` — personality prediction utilities.
- `demo_talking_video.py`, `test_talking_video.py` — small demo/test scripts.
- `requirements_talking_video.txt`, `requirements_personality.txt` — Python dependency lists.

Quick summary
- Language: Python (main), plus a small TypeScript/React frontend (`App.tsx`, `index.tsx`) and a `package.json`/`tsconfig.json` (likely a small web UI or demo).
- Intended tasks: Generate a talking head video from a single image + audio; personality prediction from audio; utilities to download model weights and prepare environments.

Prerequisites
- OS: Linux/macOS recommended. Windows support may require additional steps (`setup_windows.py` and `WINDOWS_SETUP.md`).
- Python: 3.10+ (virtualenv/venv recommended). A pre-existing venv can be found at `timetwin_env/` in this repo.
- GPU: A CUDA-capable GPU is recommended for reasonable performance when running SadTalker/Wav2Lip models. CPU-only runs are possible but slow.
- Node (optional): `node` and `npm`/`pnpm` if you want to run the TypeScript/React demo (`index.tsx`, `App.tsx`).

Installation (Python)
1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install Python dependencies for talking video functionality:

```bash
python -m pip install -r requirements_talking_video.txt
```

3. (Optional) Install personality predictor dependencies:

```bash
python -m pip install -r requirements_personality.txt
```

4. If you prefer, use the provided `timetwin_env/` virtual environment. Activate it with:

```bash
source timetwin_env/bin/activate
```

Model downloads
- Many scripts expect model weights to be present. Use the included download helpers:

```bash
python download_sadtalker_direct.py
python download_sadtalker_models.py
```

- If the downloads fail or you prefer manual download, see `SadTalker/` and `setup_sadtalker_manual.py` for instructions and expected weight filenames/locations.

Usage examples
- Basic talking-video demo (uses default image and audio if present):

```bash
python demo_talking_video.py
```

- Create a talking video from `Photo1.jpg` and a text-to-speech/wav file:

```bash
# Generate or place an audio file (wav) in repo root, e.g. hello.wav
python talking_video_simple.py --image Photo1.jpg --audio hello.wav --out output/Photo1_talking.mp4
```

- Run the SadTalker demo (if models are downloaded):

```bash
python sadtalker_working.py --image Photo1.jpg --audio hello.wav --out output/sadtalker_demo.mp4
```

- Personality prediction example:

```bash
python personality_predictor_simple.py --audio output/Hello\ there.\ Nice\ to\ meet\ you.wav
```

Tips and notes
- File paths: many scripts assume the repo root as the working directory. Run commands from the repo root to avoid path issues.
- GPU/CPU: For best results and reasonable run times, use a machine with an NVIDIA GPU and CUDA drivers installed. Without CUDA, expect long processing times or memory issues.
- Virtual environment: Use a fresh virtual environment to avoid package conflicts.
- Windows: Follow `WINDOWS_SETUP.md` and `setup_windows.py` if running on Windows. Some prebuilt wheels in `timetwin_env/Lib/site-packages/` appear Windows-specific.

Troubleshooting
- Missing model files: Re-run the download scripts or inspect `SadTalker/` and `output/` for expected filenames. Check script logs for exact paths.
- Dependency errors: Recreate the venv and re-run `pip install -r requirements_talking_video.txt`. If you see binary wheel errors, ensure `pip` and `wheel` are up-to-date: `python -m pip install --upgrade pip wheel`.
- CUDA errors: Ensure compatible CUDA drivers and that `torch` is installed with the correct CUDA version. See PyTorch install guide for matching `torch`+CUDA.
- Audio issues: Use 16-bit PCM WAV files at standard sample rates (16 kHz or 22.05/44.1 kHz). If TTS output isn't compatible, re-export to WAV using `ffmpeg`.

Project layout (selected)
- `App.tsx`, `index.tsx`, `package.json`, `tsconfig.json` — small web UI/demo (TypeScript + React + Vite). Use `npm install` and `npm run dev` in the repo root if you want to run it.
- `SadTalker/` — SadTalker model code and related weights (not necessarily complete in repo; may be downloaded by helper scripts).
- `output/` — Example outputs and generated media.

Next steps and improvements
- Add a single unified CLI that wraps the different demo scripts and normalizes arguments.
- Provide a small `Makefile` or `scripts/bootstrap.sh` to automate environment creation and model downloads.
- Add unit tests or smoke tests that validate a minimal end-to-end run using tiny models or mocked model files.

License & Credits
- This repository aggregates code from multiple projects and research repositories (SadTalker, Wav2Lip, etc.). Respect and follow the original licenses for those components. If you redistribute or publish derivatives, include the original attribution and license files.

Last updated: 2025-09-13


