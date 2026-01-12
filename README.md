# SAM-AUDIO Sound Isolator

A Python GUI application for isolating specific sounds (vocals, music, etc.) from video files using the SAM-AUDIO model.

## Features
- **User-Friendly GUI**: Easy to load video and select sounds.
- **Smart Chunking**: Processes long videos in 30-second segments.
- **Auto-Resampling**: Handles audio formats automatically.
- **Preset & Custom Prompts**: Choose "Vocals", "Music" or type your own.

## Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/sound-isolator.git
    cd sound-isolator
    ```

2.  **One-Click Setup**:
    Double-click **`setup.bat`**.
    
    This script will automatically:
    - Create a virtual environment (`sound_isolator`).
    - Install all dependencies from `requirements.txt`.

    *Alternatively, you can manually run `python -m venv sound_isolator`, activate it, and run `pip install -r requirements.txt`.*

4.  **FFMPEG**:
    Ensure [FFMPEG](https://ffmpeg.org/download.html) is installed and added to your system PATH.
    -   *Compatibility*: **Version 4.0 or newer** is recommended (tested with 5.x/6.x).
    -   *Build Type*: **"Static"** or **"Full"** builds are easiest (they contain everything in one file). "Shared" builds work too but require correct DLL placement.
    -   Basic flags used: `-i`, `-vn`, `-acodec pcm_s16le`, `-c:v copy`.

## Usage

1.  **Run the App**:
    Double-click `start_app.bat` (if on Windows) or run:
    ```bash
    python sound_isolator_app.py
    ```

2.  **Load Model**: Click "Load Model" to initialize SAM-AUDIO.
3.  **Process**: Select your video and target sound, then click "Isolate Sound".

## Models
The application requires model checkpoints to run. Create a `models` folder and download models into subfolders (e.g., `models/sam-audio-large/`).

**Download Links:**
- [SAM-Audio Small](https://huggingface.co/facebook/sam-audio-small)
- [SAM-Audio Base](https://huggingface.co/facebook/sam-audio-base)
- [SAM-Audio Large](https://huggingface.co/facebook/sam-audio-large)

## Requirements
- Python 3.10+ (Tested on 3.12)
- CUDA-capable GPU recommended (but works on CPU)

> **Note on First Run**:
> When you load the model for the first time, it will automatically download several dependencies:
> - **ImageBind** (~4GB)
> - **T5 Text Encoder** (~1GB)
>
> This is normal. These files are saved to `.checkpoints` or your system cache and reused for all models (Small/Base/Large), so you won't need to download them again.
