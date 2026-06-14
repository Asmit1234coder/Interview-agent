📖 Project Context: Live Vision Agent
This project is a real‑time multimodal AI agent that connects microphone audio and webcam video to Google Gemini’s Live API. The agent listens, sees, and responds out loud in a conversational way.

🎯 What the project does
Captures microphone audio and streams it to Gemini.

Captures webcam frames (JPEG) and streams them once per second.

Receives real‑time audio responses from Gemini and plays them back through the speaker.

Prints transcriptions of the model’s spoken output.

Runs all tasks concurrently using asyncio (mic, camera, response handling, playback).

The result is an interactive agent that can see what you show it and talk back to you in real time.

📂 Project Structure
Code
live-vision-agent/
├── config.py        # Central constants (API key, audio/video settings, system prompt)
├── webcam.py        # Camera class to capture JPEG frames
├── audio.py         # AudioIo class to manage mic & speaker streams
├── agents.py        # LiveVisionAgent orchestrating mic, camera, Gemini API
├── main.py          # Entry point to run the agent
└── .env             # Holds GEMINI_API_KEY and other secrets
⚙️ Components
config.py  
Holds configuration constants: API key, model name, voice, audio format, sample rates, frame size, JPEG quality, and system prompt.

webcam.py  
Defines Camera class. Opens webcam, reads frames, converts to RGB, compresses to JPEG, returns bytes.

audio.py  
Defines AudioIo class. Manages microphone and speaker streams using PyAudio. Provides methods to open mic, open speaker, and close streams.

agents.py  
Defines LiveVisionAgent.

Creates a Gemini client with your API key.

Sets up LIVE_CONFIG with audio response modality, system prompt, and voice config.

Runs four async tasks:

_stream_microphone → sends mic audio upstream.

_stream_camera → sends JPEG frames upstream.

_receive_responses → receives model responses, queues audio, prints transcript.

_playback_audio → plays audio chunks through speaker.

main.py  
Entry point that instantiates LiveVisionAgent and calls run() inside asyncio.run.

🚧 Current Work
You’ve already built all core modules.

The only remaining step is aligning VoiceConfig with the exact schema of your installed google-genai SDK (field names differ between versions).

Once fixed, the agent will run end‑to‑end: mic + camera → Gemini → audio playback.