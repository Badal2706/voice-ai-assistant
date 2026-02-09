# 🎙️ Local Voice AI Assistant

A fully offline, privacy-focused voice-to-voice AI assistant that runs completely on your device with low latency.

## Features

- 🚀 **100% Local** - No data sent to cloud services
- 🔒 **Privacy First** - All processing happens on your device
- ⚡ **Low Latency** - Fast response times using optimized models
- 🎤 **Voice Activity Detection** - Automatically detects when you stop speaking
- 💬 **Conversational Memory** - Maintains context across the conversation

## Tech Stack

- **Speech-to-Text**: Faster Whisper (OpenAI Whisper optimized)
- **LLM**: Ollama (Llama 3.2 or other local models)
- **Text-to-Speech**: pyttsx3 (offline TTS)
- **Audio Processing**: PyAudio + NumPy

## Installation

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.com) installed and running locally
- Microphone access

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/voice-ai-assistant.git
   cd voice-ai-assistant
   
2. Install dependencies:Install dependencies:
   ```bash
    pip install -r requirements.txt
   
3. Download a Whisper model (automatic on first run) and ensure Ollama has your chosen model:
    ```bash
    ollama pull llama3.2
   
### Usage

- Run the assistant:
    ```bash
    python voice_assistant.py

**Controls:**
- Speak naturally - the assistant listens automatically (VAD mode)
- Press Q anytime to quit
- Say "goodbye", "bye", or "exit" to end the conversation gracefully

**Configuration**

- Edit these parameters in voice_assistant.py:

| Parameter          | Default      | Description                                  |
| ------------------ | ------------ | -------------------------------------------- |
| `whisper_model`    | `"base"`     | Model size: tiny, base, small, medium, large |
| `ollama_model`     | `"llama3.2"` | Any Ollama model you have installed          |
| `device`           | `"cpu"`      | Use `"cuda"` for GPU acceleration            |
| `compute_type`     | `"int8"`     | `"float16"` for GPU, `"int8"` for CPU        |
| `silence_duration` | `2.0`        | Seconds of silence before stopping recording |

**Project Structure**

    voice-ai-assistant/
    ├── voice_assistant.py    # Main application code
    ├── requirements.txt      # Python dependencies
    └── README.md            # This file

**How It Works**

1. Recording: Uses Voice Activity Detection (VAD) to start/stop recording based on your voice
2. Transcription: Faster Whisper converts speech to text locally
3. LLM Processing: Ollama generates a response using your chosen local model
4. Speech: pyttsx3 speaks the response using your system's TTS engine
5. Loop: Repeats until you say goodbye or press Q

**Requirements**

- RAM: 4GB+ recommended (depends on Whisper model size)
- Storage: ~2GB for models
- CPU: Modern multi-core processor (GPU optional)

**License**

    MIT License - Feel free to use and modify!

**Contributing**

    Pull requests welcome! This is a great base for building custom voice assistants.

**Acknowledgments**

    Faster Whisper for efficient STT
    Ollama for easy local LLM hosting
    pyttsx3 for offline TTS

### Built with ❤️ by Badal Patel