---
title: "Build a Voice AI Assistant: Speech-to-Speech Conversational AI"
description: "Build a voice-enabled AI assistant using Whisper for speech recognition, GPT-4o-mini for responses, and TTS for natural voice output. Supports wake word detection and conversation memory."
date: "2026-03-10"
slug: "voice-ai-assistant"
level: "Intermediate"
time: "4–6 hours"
stack: "Python, OpenAI API (Whisper + GPT + TTS), sounddevice, pygame"
keywords: ["voice AI assistant Python", "Whisper speech recognition", "text to speech AI Python"]
---

## Project Overview

A voice-enabled conversational assistant that listens for speech, transcribes with Whisper, generates responses with GPT-4o-mini, and speaks the reply using OpenAI TTS. Maintains conversation history for natural back-and-forth dialogue.

---

## Learning Goals

- Record and process microphone audio in Python
- Integrate Whisper STT, GPT-4o-mini, and TTS in a pipeline
- Manage conversation state for multi-turn voice dialogue
- Handle audio playback and latency optimization

---

## Architecture

```
Microphone input → record audio
        ↓
Whisper API → transcribed text
        ↓
GPT-4o-mini + conversation history → response text
        ↓
OpenAI TTS → audio bytes
        ↓
Speaker output (pygame)
```

---

## Implementation

### Step 1: Setup

```bash
pip install openai sounddevice numpy scipy pygame
```

On macOS you may also need: `brew install portaudio`

### Step 2: Audio Recorder

```python
# recorder.py
import io
import numpy as np
import sounddevice as sd
from scipy.io import wavfile


SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 0.01   # RMS below this = silence
SILENCE_DURATION = 1.5     # seconds of silence to stop recording
MAX_DURATION = 30          # max recording seconds


def record_until_silence() -> bytes:
    """Record audio until the user stops speaking."""
    print("🎤 Listening...")
    chunks = []
    silence_frames = 0
    frames_per_check = int(SAMPLE_RATE * 0.1)  # check every 100ms
    silence_limit = int(SILENCE_DURATION / 0.1)
    min_speech_frames = int(0.5 / 0.1)  # at least 500ms of speech before we check for silence
    speech_detected = 0

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32") as stream:
        while len(chunks) * frames_per_check / SAMPLE_RATE < MAX_DURATION:
            frame, _ = stream.read(frames_per_check)
            chunks.append(frame.copy())
            rms = np.sqrt(np.mean(frame ** 2))

            if rms > SILENCE_THRESHOLD:
                silence_frames = 0
                speech_detected += 1
            elif speech_detected > min_speech_frames:
                silence_frames += 1
                if silence_frames >= silence_limit:
                    break

    audio = np.concatenate(chunks, axis=0)
    audio_int16 = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, SAMPLE_RATE, audio_int16)
    buf.seek(0)
    return buf.read()


def play_audio(audio_bytes: bytes):
    """Play audio bytes through speakers."""
    import pygame
    pygame.mixer.init()
    sound = pygame.mixer.Sound(io.BytesIO(audio_bytes))
    sound.play()
    # Wait for playback to finish
    while pygame.mixer.get_busy():
        pygame.time.wait(50)
```

### Step 3: Voice Pipeline

```python
# assistant.py
import io
from openai import OpenAI
from recorder import record_until_silence, play_audio

client = OpenAI()

SYSTEM_PROMPT = """You are a helpful voice assistant. Keep responses concise (2-3 sentences max) since they will be spoken aloud. Be conversational and natural."""


def transcribe(audio_bytes: bytes) -> str:
    """Convert speech to text using Whisper."""
    buf = io.BytesIO(audio_bytes)
    buf.name = "audio.wav"
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=buf,
        response_format="text",
    )
    return transcript.strip()


def generate_response(text: str, history: list[dict]) -> str:
    """Generate a conversational response."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-10:])  # Keep last 10 turns
    messages.append({"role": "user", "content": text})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=200,
        temperature=0.7,
    )
    return response.choices[0].message.content


def speak(text: str, voice: str = "nova"):
    """Convert text to speech and play it."""
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,  # alloy, echo, fable, onyx, nova, shimmer
        input=text,
        response_format="mp3",
    )
    audio_bytes = response.content
    play_audio(audio_bytes)


def run_assistant():
    """Main voice conversation loop."""
    print("🤖 Voice Assistant ready. Press Ctrl+C to quit.")
    history = []

    while True:
        try:
            # Record
            audio = record_until_silence()

            # Transcribe
            user_text = transcribe(audio)
            if not user_text or len(user_text) < 3:
                print("(no speech detected)")
                continue
            print(f"You: {user_text}")

            # Generate response
            response = generate_response(user_text, history)
            print(f"Assistant: {response}")

            # Update history
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": response})

            # Speak
            speak(response)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    run_assistant()
```

### Step 4: Enhanced Version with Commands

```python
# enhanced_assistant.py
from assistant import client, transcribe, generate_response, speak, record_until_silence

WAKE_COMMANDS = {"quit", "exit", "goodbye", "stop"}
SPECIAL_COMMANDS = {
    "clear history": lambda h: h.clear() or "History cleared.",
    "what time is it": lambda _: __import__("datetime").datetime.now().strftime("It's %I:%M %p."),
}


def run_enhanced():
    print("🤖 Enhanced Voice Assistant ready.")
    history = []

    while True:
        try:
            audio = record_until_silence()
            user_text = transcribe(audio)

            if not user_text:
                continue

            print(f"You: {user_text}")
            lower = user_text.lower().strip()

            # Wake commands
            if any(cmd in lower for cmd in WAKE_COMMANDS):
                speak("Goodbye!")
                break

            # Special commands
            handled = False
            for trigger, handler in SPECIAL_COMMANDS.items():
                if trigger in lower:
                    response = handler(history)
                    print(f"Assistant: {response}")
                    speak(response)
                    handled = True
                    break

            if not handled:
                response = generate_response(user_text, history)
                print(f"Assistant: {response}")
                history.append({"role": "user", "content": user_text})
                history.append({"role": "assistant", "content": response})
                speak(response)

        except KeyboardInterrupt:
            speak("Goodbye!")
            break


if __name__ == "__main__":
    run_enhanced()
```

### Step 5: Run

```bash
python assistant.py
# Or the enhanced version:
python enhanced_assistant.py
```

---

## Extension Ideas

1. **Wake word detection** — use Porcupine or snowboy for "Hey Assistant" activation
2. **Multi-language** — detect spoken language with Whisper and respond in kind
3. **Smart home integration** — control Home Assistant via voice commands
4. **Calendar integration** — check/add calendar events by voice
5. **Streaming TTS** — stream audio as it generates for lower latency

---

## What to Learn Next

- **OpenAI audio APIs** → [OpenAI API Complete Guide](/blog/openai-api-complete-guide/)
- **AI agents** → [AI Agent Fundamentals](/blog/ai-agent-fundamentals/)
