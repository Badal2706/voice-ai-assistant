import asyncio
import pyaudio
import wave
import numpy as np
from faster_whisper import WhisperModel
import ollama
import pyttsx3
from collections import deque
import keyboard
import time


class VoiceAssistant:
    def __init__(
            self,
            whisper_model="base",
            ollama_model="llama3.2",
            device="cpu",
            compute_type="int8",
            system_prompt="You are a helpful AI assistant. Provide clear, concise answers."
    ):
        # Initialize Faster Whisper
        print("Loading Whisper model...")
        self.whisper = WhisperModel(
            whisper_model,
            device=device,
            compute_type=compute_type
        )

        # Initialize TTS - create new engine each time
        print("Initializing TTS...")
        self.tts_rate = 175

        # Ollama model
        self.ollama_model = ollama_model
        self.system_prompt = system_prompt

        # Audio settings
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.audio = pyaudio.PyAudio()

        # Conversation history
        self.conversation_history = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]

        print("Assistant ready!")

    def record_audio(self, duration=5, filename="temp_audio.wav"):
        """Record audio from microphone"""
        print(f"Recording for {duration} seconds...")

        stream = self.audio.open(
            format=self.FORMAT,            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )

        frames = []
        for _ in range(0, int(self.RATE / self.CHUNK * duration)):
            data = stream.read(self.CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()

        # Save to file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        print("Recording complete!")
        return filename

    def record_with_vad(self, filename="temp_audio.wav", silence_duration=2.0):
        """Record with Voice Activity Detection - stops after silence"""
        print("Listening... (speak now, will stop after silence)")

        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )

        frames = []
        silent_chunks = 0
        silence_threshold = int(self.RATE / self.CHUNK * silence_duration)
        started = False

        while True:
            data = stream.read(self.CHUNK)
            frames.append(data)

            # Simple VAD based on volume
            audio_data = np.frombuffer(data, dtype=np.int16)
            volume = np.abs(audio_data).mean()

            if volume > 100:  # Adjust threshold as needed
                silent_chunks = 0
                started = True
            else:
                if started:
                    silent_chunks += 1

            if started and silent_chunks > silence_threshold:
                break

        stream.stop_stream()
        stream.close()

        # Save to file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        print("Recording complete!")
        return filename

    def transcribe(self, audio_file):
        """Convert speech to text using Faster Whisper"""
        print("Transcribing...")
        segments, info = self.whisper.transcribe(
            audio_file,
            beam_size=5,
            language="en"
        )

        text = " ".join([segment.text for segment in segments])
        print(f"You said: {text}")
        return text.strip()

    def is_goodbye(self, text):
        """Check if user is saying goodbye"""
        goodbye_phrases = [
            'goodbye', 'good bye', 'bye', 'bye bye',
            'see you', 'talk to you later', 'ttyl',
            'gotta go', 'have to go', 'exit', 'quit',
            'stop', 'end conversation', 'that\'s all'
        ]
        text_lower = text.lower().strip()
        return any(phrase in text_lower for phrase in goodbye_phrases)

    def get_llm_response(self, user_input):
        """Get response from Ollama LLM"""
        print("Thinking...")

        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })

        # Get response from Ollama
        response = ollama.chat(
            model=self.ollama_model,
            messages=self.conversation_history
        )

        assistant_response = response['message']['content']

        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_response
        })

        # Keep only last 10 messages (plus system prompt)
        if len(self.conversation_history) > 11:
            # Keep system prompt and last 10 messages
            self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-10:]

        print(f"Assistant: {assistant_response}")
        return assistant_response

    def speak(self, text):
        """Convert text to speech - creates fresh engine each time"""
        print("Speaking...")
        try:
            # Create a new engine instance for each speech
            engine = pyttsx3.init()
            engine.setProperty('rate', self.tts_rate)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine  # Clean up
        except Exception as e:
            print(f"TTS Error: {e}")

    def run_conversation(self, use_vad=True):
        """Main conversation loop"""
        print("\n=== Voice Assistant Started ===")
        print("Press 'q' to quit or say 'goodbye' to end\n")

        while True:
            try:
                # Check for quit
                if keyboard.is_pressed('q'):
                    print("Exiting...")
                    break

                # Record audio
                if use_vad:
                    audio_file = self.record_with_vad()
                else:
                    audio_file = self.record_audio(duration=5)

                # Transcribe
                user_text = self.transcribe(audio_file)

                if not user_text:
                    print("No speech detected, try again...")
                    continue

                # Check for goodbye
                if self.is_goodbye(user_text):
                    print("Goodbye detected, ending conversation...")
                    response = self.get_llm_response(user_text)
                    self.speak(response)
                    print("\n=== Conversation Ended ===")
                    break

                # Get LLM response
                response = self.get_llm_response(user_text)

                # Speak response
                self.speak(response)

                print("\n" + "=" * 50 + "\n")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue

    def cleanup(self):
        """Cleanup resources"""
        self.audio.terminate()


# Example usage
if __name__ == "__main__":
    # Custom system prompt
    system_prompt = """You are a helpful and friendly AI assistant. 
    Provide clear, concise answers. Keep responses brief and conversational not unnecessarily long answer.
    Don't talk about your training data and answer like human.
    If asked about your capabilities, explain that you can have voice conversations."""

    # Initialize assistant
    assistant = VoiceAssistant(
        whisper_model="base",  # Options: tiny, base, small, medium, large
        ollama_model="llama3.2",  # Llama 3.2 model
        device="cpu",  # or "cuda" for GPU
        compute_type="int8",  # int8 for CPU, float16 for GPU
        system_prompt=system_prompt
    )

    try:
        # Run the conversation loop
        assistant.run_conversation(use_vad=True)
    finally:
        assistant.cleanup()